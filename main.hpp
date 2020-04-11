/*
 * main.hpp
 *
 *  Created on: 4 jan. 2020
 *      Author: Joeran Bosma
 */

#ifndef MAIN_HPP_
#define MAIN_HPP_

#include <iostream>
#include <thread>
#include <list>
#include <random>
#include <chrono>
// after: http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "Eigen/LU"
#include <math.h>
#include <mutex>
#include "stats.hpp"
using namespace Eigen;
using namespace std;

class Worker {
public:
	Worker(MatrixXd w, VectorXd h, const int N,
			const int iterations = 100000, const int num_threads = 3):
		_w(w), _h(h), _N(N),
		_iterations(iterations), _num_threads(num_threads) { }

	~Worker() {
		cout << "~Destructor" << endl;
	}

	Stats get_stats() {
		sample();
		calc_stats();
		combine_stats();
		return final_stats;
	}

	list<MatrixXd> get_samples_list() { return samples_list; }

	void sample() {
		// sample iterations distributed over multiple threads
		int its_per_thread = _iterations / _num_threads;

		cout << "Starting sampling (N=" << _N << ")..." << endl;
		auto start = chrono::steady_clock::now();

		for (int i = 0; i < _num_threads; i++) {
			threads.push_back( thread(&Worker::MCMCsampler, this, its_per_thread, ref(_m)) );
		}
		for (list<thread>::iterator it = threads.begin(); it != threads.end(); it ++) {
			(*it).join();
		}

		threads.clear();

		auto end = chrono::steady_clock::now();
		cout << "Elapsed time: "
				<< chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.
				<< "s" << endl;
	}

	void calc_stats() {
		cout << "Starting stats..." << endl;
		auto start = chrono::steady_clock::now();

		for (list<MatrixXd>::iterator it = samples_list.begin(); it != samples_list.end(); it ++) {
			cout << "Start calculating stats of " << (*it).rows() << "x" << (*it).cols() << endl;
			threads.push_back( thread(&Worker::MCMCstats, this, ref( *it ), ref(_m) ) );
		}
		for (list<thread>::iterator it = threads.begin(); it != threads.end(); it ++) {
			(*it).join();
		}

		threads.clear();

		auto end = chrono::steady_clock::now();
		cout << "Elapsed time: "
				<< chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.
				<< "s" << endl;
	}

	void test(const Eigen::Ref<const MatrixXd> &v) {
		cout << "Test!" << endl;
		cout << v << endl;
	}

	void combine_stats() {
		// combine list of Stats objects into single Stats object
		// NOTE: ASSUMES EQUAL WEIGHT FOR EACH ENTRY! (i.e. equal number of steps of MCMC)
		cout << "Starting combining..." << endl;
		auto start = chrono::steady_clock::now();

		Stats comb(results.front());

		for (list<Stats>::iterator it = ++results.begin(); it != results.end(); it ++) {
			comb.addS( (*it).getS() );
			comb.addSS( (*it).getSS() );
		}

		// divide by number of elements to get mean instead of sum
		comb.setS( comb.getS() / results.size() );
		comb.setSS( comb.getSS() / results.size() );

		final_stats = comb;

		auto end = chrono::steady_clock::now();
		cout << "Elapsed time: "
				<< chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.
				<< "s" << endl;
	}

	double findE(VectorXd& s, MatrixXd& w, VectorXd& h) {
		// Calculate -E(s) for state s with couplings w and thresholds h
		// The (0, 0) at the end converts the 1x1 matrix to a double
		return -(0.5 * s.transpose() * w * s + s.transpose() * h)(0, 0);
	}

	double approxZ(int num_samples) {
		long double pstar_sum = 0;

		// random number generator
		random_device rd;
		mt19937 rng(rd());
		uniform_int_distribution<int> uni(0, 1); // inclusive range

		// initialize state
		VectorXd s(_N);

		for (int i = 0; i < num_samples; i++) {
			// generate random state
			for (int i = 0; i < _N; i++) {
				int spin = uni(rng);
				s(i) = 2 * spin - 1; // convert {0, 1} to {-1, 1}
			}
			// calculate p*(s) and add to sum
			pstar_sum += exp(-findE(s, _w, _h));
		}

		// return result
		return pstar_sum * (pow(2, _N) / num_samples);
	}


private:
	void MCMCsampler(const int iterations, mutex& m) {
		// setup
		const float burn = 0.05;
		int accepted = 0;
		MatrixXd samples(iterations, _N);

		// random number generator
		random_device rd;
		mt19937 rng(rd());
		uniform_int_distribution<int> uni(0, _N-1); // inclusive range
		uniform_real_distribution<double> acc(0, 1);

		// generate starting state
		VectorXd s(_N);
		for (int i = 0; i < _N; i++) {
			s(i) = -1;
		}

		double E = findE(s, _w, _h);
		double Enew;

		for (int it = 0; it < iterations; it++) {
			// copy state
			VectorXd snew(s);
			// flip random spin
			int idx = uni(rng);
			snew(idx) *= -1;

			// find new energy and check whether to accept new state
			Enew = findE(snew, _w, _h);
			if (Enew < E) { // accept proposed state
				s = snew;
				E = Enew;
				accepted ++;
				for (int i = 0; i < _N; i++) {
					samples(it, i) = s(i);
				}
			} else if (exp(E - Enew) > acc(rng)) {
				// also accept proposed state
				s = snew;
				E = Enew;
				accepted ++;
				for (int i = 0; i < _N; i++) {
					samples(it, i) = s(i);
				}
			} else {
				// Rejected
				for (int i = 0; i < _N; i++) {
					samples(it, i) = s(i);
				}
			}
		}

		double acceptance_ratio = (accepted + 0.0 ) / iterations * 100;
		cout << "Accepted " << acceptance_ratio << "% of samples" << endl;

		// burn samples
		int burn_num = burn * iterations;
		cout << "Burn " << burn_num << " samples" << endl;

		// add samples to shared list
		lock_guard<mutex> guard(m);
		samples_list.push_back( samples.bottomLeftCorner(samples.rows() - burn_num, _N) );
	}

	void MCMCstats(MatrixXd& samples, mutex& m) {
		//    samples, acceptance_ratio = MHMC_sampler(w, h, return_samples=True, return_stats=False, iterations=int(iterations), flip=flip, verbose=verbose)
		cout << "Calculating <s_i> and <s_i s_j>..." << endl;
		// initialize s_i with first element to specify the size of the vector
		VectorXd si(_N);
		MatrixXd ss(_N, _N);

		Stats stats(samples.colwise().mean(),
					samples.transpose() * samples / samples.rows());

		// add stats to shared list
		lock_guard<mutex> guard(m);
		results.push_back( stats );
	}

	// MCMC stuff
	MatrixXd _w;
	VectorXd _h;
	int _N;

	// Worker stuff
	int _iterations;
	int _num_threads;
	list<thread> threads;
	list<MatrixXd> samples_list;
	list<Stats> results;
	Stats final_stats;
	mutex _m;
};



#endif /* MAIN_HPP_ */

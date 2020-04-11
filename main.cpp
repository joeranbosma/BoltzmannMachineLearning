/*
 * main.cpp
 *
 *  Created on: 4 jan. 2020
 *      Author: Joeran Bosma
 *
 * This C++ project is written as extension of the Boltzmann Machine Learning notebook.
 * The link between C++ and Python is provided by Pybind11 and Eigen for NumPy/Eigen matrices.
 * Compiling the project is possible from within Python, or from the command line:
 * - Command line: python3 setup.py build
 * - Jupyter Notebook: !python3 setup.py build
 * - Python script: os.system("python3 setup.py build")
 *
 * Alternatively, this project can be compiled with the following command:
 * g++ -o main main.cpp -std=c++11 -lpthread
 * This requires the packages Eigen and Pybind11 to be installed.
 */

#include <iostream>
#include "main.hpp"
#include "pybind11/pybind11.h"
// after: http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "Eigen/LU"
#include "stats.hpp"
using namespace Eigen;
using namespace std;

int main() {
//	Eigen::setNbThreads(4);

    // Test implementation of sampler
	const int N = 160;
	MatrixXd w(N, N);
	VectorXd h(N);

	// Use random number generator to generate random couplings and external field
	random_device rd;
	mt19937 rng(rd());
	uniform_real_distribution<double> acc(0, 1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			w(i, j) = acc(rng);
		}
		h(i) = acc(rng);
		w(i, i) = 0; // set coupling to self to zero
	}

	Worker worker(w, h, N);
	Stats stats = worker.get_stats();

	return 0;
}

// Implement glue between C++ and Python using Pybind11
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(Worker, m) {
    // Make C++ functions accessible from Python
	py::class_<Worker>(m, "Worker")//, py::dynamic_attr())
			.def(py::init<Eigen::MatrixXd, Eigen::VectorXd, const int,
					const int, const int>())
			.def("test", &Worker::test, py::arg())//.noconvert() )
			.def("get_stats", &Worker::get_stats)
			.def("get_samples_list", &Worker::get_samples_list)
			.def("sample", &Worker::sample)
			.def("findE", &Worker::findE, py::arg(), py::arg(), py::arg())
			.def("approxZ", &Worker::approxZ, py::arg())
	.def("__repr__", [](const Worker& a) { return "<Worker on .. threads>";});

	m.doc() = "pybind11 Worker"; // optional module docstring

    // Also provide an interface for the Stats class, which is used to bundle the results
    // of the mean firing rates <s_i> and mean firing rate products <s_i * s_j>
	py::class_<Stats>(m, "Stats")//, py::dynamic_attr())
			.def(py::init<>())
			.def("getS", &Stats::getS)
			.def("getSS", &Stats::getSS)
	.def("__repr__", [](const Stats& a) { return "<Stats object>";});

	m.doc() = "pybind11 Stats"; // optional module docstring
}


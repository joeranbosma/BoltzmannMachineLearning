/*
 * stats.hpp
 *
 *  Created on: 5 jan. 2020
 *      Author: Joeran Bosma
 */

#ifndef STATS_HPP_
#define STATS_HPP_

#include "pybind11/eigen.h"
#include "Eigen/LU"
using namespace Eigen;

class Stats {
public:
	Stats() {}
	Stats(VectorXd si, MatrixXd ss): _si(si), _ss(ss) {}
//	Stats(const Stats& other): _si(other._si), _ss(other._ss) {}
	VectorXd getS() { return _si; }
	MatrixXd getSS() { return _ss; }
	void addS(VectorXd s) { _si += s; }
	void addSS(MatrixXd ss) { _ss += ss; }
	void setS(VectorXd s) { _si = s; }
	void setSS(MatrixXd ss) { _ss = ss; }

private:
	VectorXd _si;
	MatrixXd _ss;
};


#endif /* STATS_HPP_ */

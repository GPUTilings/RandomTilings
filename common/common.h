#ifndef SRC_COMMON_UTIL_H_
#define SRC_COMMON_UTIL_H_

/* common.h
 *
 *  Created on: 2017
 *      Author: David, Ananth
 *
 * This header contains some utility functions.
 *
 */

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include "cl.hpp"
#else
#include <CL/cl.hpp>
#endif

#include <cmath>
#include <string>
#include <climits>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>
#include <stack>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <climits>
#include <random>
#include <chrono>

#define infty INT_MAX
#define sqrt3 1.7320508075688772935274

typedef std::vector<int> domain;
typedef std::vector<int> heightfunc;
typedef std::vector<int> tiling;

// Queries and prints your OpenCL info.
void PrintOpenCLInfo();

// Saves an MxN matrix to the disk with name filename.
void SaveMatrix(std::vector<double> &tiling, int M, int N, std::string filename);
void SaveMatrix(std::vector<int> &tiling, int M, int N, std::string filename);

// Prints an MxN matrix to the console.
void PrintMatrix(std::vector<int> &mat, int M, int N);

// Adds/removes a zero padding around an MxN matrix v.
void PadMatrix(std::vector<int> &v, int M, int N);
void UnPadMatrix(std::vector<int> &v, int M, int N);

// The above functions for square matrices.
void PadMatrix(std::vector<int> &v);
void UnPadMatrix(std::vector<int> &v);
void PrintMatrix(std::vector<int> &mat);
void SaveMatrix(std::vector<int> &mat, std::string filename);
void SaveMatrix(std::vector<double> &mat, std::string filename);

// Loads an integer matrix filename from the disk.
std::vector<int> LoadMatrix(std::string filename);

// Loads an OpenCL program.
cl::Program LoadCLProgram(cl::Context context, std::vector<cl::Device> devices, std::string input);

// We should get rid of these at some point.
struct incheightPriority {
	bool operator()(std::pair<std::pair<int,int>, int> a, std::pair<std::pair<int,int>, int> b) const {
			return a.second < b.second;
	}
};

struct decheightPriority {
	bool operator()(std::pair<std::pair<int,int>, int> a, std::pair<std::pair<int,int>, int> b) const {
		return a.second > b.second;
	}
};

#endif /* SRC_COMMON_UTIL_H_ */

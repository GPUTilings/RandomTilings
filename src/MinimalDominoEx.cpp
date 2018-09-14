/*
 * minimaldomino_example.cpp
 * 
 *
 *  Created on: Dec 16, 2017
 *      Author: David, Ananth
 *
 * Here is a minimal example that randomly tiles
 * a rectangle with dominos.
 */

#include "./common/common.h"
#include "./Domino/DominoTiler.h"

int main() {
	std::cout<<"Running Domino, Basic Example."<<std::endl;
	
	//Standard OpenCL set up code.
	cl::Context context(CL_DEVICE_TYPE_DEFAULT);
	std::string sinfo;
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	devices[0].getInfo(CL_DEVICE_NAME, &sinfo); // Check which GPU you use!
	std::cout<<"Created context using: "<<sinfo<<std::endl;
	cl::CommandQueue queue(context);

	cl_int err = 0;

	// It's usually easier to draw a domain, and then let Thurston's algorithm
	// do the work to generate a tiling. But for simplicity, I'll write out the tiling
	// explicitly.
	//
	// See figure 9 in the paper to understand the following tiling.
	tiling T = {0,0,0,0,0,0,0,0,0,0, 
				0,0,0,0,0,0,0,0,0,0,
				0,0,0,0,2,0,0,0,0,0,
				0,0,8,4,1,8,4,0,0,0,
				0,0,0,8,12,4,0,0,0,0,
				0,0,8,4,2,8,4,0,0,0,
				0,0,0,0,1,0,0,0,0,0,
				0,0,0,0,0,0,0,0,0,0,
				0,0,0,0,0,0,0,0,0,0,
				0,0,0,0,0,0,0,0,0,0};
				
	// draw the tiling, just to check
	DominoTiler::TilingToSVG(T, "./ExampleTilings/MinimalDomino/StartTiling.svg");

	// make the tiler, which loads the kernel source and compiles it for the device.
	DominoTiler D(context, queue, devices, "./src/domino/dominokernel.cl", err);
	
	// load the tinyMT parameters, for random number generators
	D.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", T.size()/2);

	// Random walk for 100 steps with seed 34893.
	D.Walk(T, 100, 34893);

	// Draw and print the result to the std out
	DominoTiler::TilingToSVG(T, "./ExampleTilings/MinimalDomino/RandomTiling.svg");
	PrintMatrix(T);
}


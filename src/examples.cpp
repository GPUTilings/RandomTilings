
/*
 * examples.cpp
 *
 *  Created on: Dec 16, 2017
 *      Author: David, Ananth
 *
 * This contains many examples.
 */


#define __CL_ENABLE_EXCEPTIONS
#include "./common/common.h"
#include "./Domino/DominoTiler.h"
#include "./TriangleDimer/TriangleDimerTiler.h"
#include "./RectTriangle/RectTriangleTiler.h"

#include <chrono>


void DominoBasicEx(); // A minimal example.
void AztecDiamondCFTP(int N); // Tiles an Aztec Diamond of order N, using coupling from the past.
void TriangleDimerEx(int N, int M, int Nsteps); // Dimers on the triangular lattice.
void RectTriangleEx(int N, int Nsteps); // A basic example for Rectangle-Triangle tilings of a hexagon.


int main() {
	//AztecDiamondCFTP(10);
    TriangleDimerEx(5,2,100);
    RectTriangleEx(5,500);
}


void DominoBasicEx() {
	std::cout<<"Running Domino, Basic Example."<<std::endl;

	//Standard OpenCL set up code.
	PrintOpenCLInfo(); // Look at what devices are available
	cl::Context context(CL_DEVICE_TYPE_DEFAULT);
	std::string sinfo;
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	devices[0].getInfo(CL_DEVICE_NAME, &sinfo); // Check which GPU you use!
	std::cout<<"Created context using: "<<sinfo<<std::endl;
	cl::CommandQueue queue(context);

	cl_int err = 0;

	tiling T = {0,0,0,0,0,0,0,0,0,0, // see figure 9 in the paper.
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
	DominoTiler::TilingToSVG(T, "./examples/BasicDomino/StartTiling.svg");

	// make the tiler, which loads the kernel source and compiles it for the device.
	DominoTiler D(context, queue, devices, "./src/domino/dominokernel.cl", err);
	// load the tinyMT parameters
	D.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", T.size()/2);

	// Random walk for 100 steps with seed 34893.
	D.Walk(T, 100, 34893);

	// Draw and print the result
	DominoTiler::TilingToSVG(T, "./examples/BasicDomino/RandomTiling.svg");
	PrintMatrix(T);
}


void AztecDiamondCFTP(int N) {
	std::cout<<"Running Aztec Diamond CFTP."<<std::endl;

	//Standard OpenCL set up code.
	//PrintOpenCLInfo(); // Look at what devices are available
	cl::Context context(CL_DEVICE_TYPE_DEFAULT);
	std::string sinfo;
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	devices[0].getInfo(CL_DEVICE_NAME, &sinfo); // Check which GPU you use!
	std::cout<<"Created context using: "<<sinfo<<std::endl;
	cl::CommandQueue queue(context);

	cl_int err = 0;

	domain d = DominoTiler::AztecDiamond(N);
	tiling tMax = DominoTiler::MaxTiling(d);
	tiling tMin = DominoTiler::MinTiling(d);

	DominoTiler D(context, queue, devices, "./src/domino/dominokernelCFTP.cl", err); // use the CFTP kernel!
	D.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", tMax.size()/2);

	std::mt19937 mt_rand(345903);


	// Coupling from the past, with step sizes increasing by powers of two.
	std::vector<long> steps; std::vector<long> seeds;

	steps.push_back(128); seeds.push_back(mt_rand());


	while ( true ) {
		tiling top = tMax;
		tiling bottom = tMin;

		// run the random walks with the list of steps and seeds
		D.Walk(bottom, steps, seeds);
		D.Walk(top, steps, seeds);

		// check if coupled
		bool coupled = true;
		for (int i = 0; i < top.size(); i++)
			coupled = coupled & (top[i] == bottom[i]);

		if ( coupled ) {
			DominoTiler::TilingToSVG(top, "./examples/AztecDiamondCFTP/RandomTiling.svg");
			break;
		}

		steps.insert(steps.begin(),steps[steps.size()-1]*2); seeds.insert(seeds.begin(),mt_rand());
	}
}

void TriangleDimerEx(int N, int M, int Nsteps) {
    // Tiles an equillateral triangle of side length N, oriented point downward, with a equillateral triangle of size M cut out of the top left corner.
    
    std::cout<<"Running basic Dimer on Triangular Lattice example."<<std::endl;
    auto start = std::chrono::steady_clock::now();
    
    //Standard OpenCL set up code.
    //PrintOpenCLInfo(); // Look at what devices are available
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    std::string sinfo;
    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    devices[0].getInfo(CL_DEVICE_NAME, &sinfo);
    cl::CommandQueue queue(context);
    cl_int err = 0;
    
    //Create domain and starting tiling and draw them.
    tiling t = TriangleDimerTiler::IceCreamCone(N,M);
    //tiling t = TriangleDimerTiler::TestButterfly(0);
    domain d = TriangleDimerTiler::TilingToDomain(t);
    SaveMatrix(d,"./examples/TriangleDimer/Domain.txt");
    SaveMatrix(t,"./examples/TriangleDimer/DimerStart.txt");
    TriangleDimerTiler::DomainToSVG(d,"./examples/TriangleDimer/Domain.svg");
    TriangleDimerTiler::DimerToSVG(t,"./examples/TriangleDimer/DimerStart.svg");
    TriangleDimerTiler::TilingToSVG(t,"./examples/TriangleDimer/TilingStart.svg");
    
    
    //Set up the mcmc.
    TriangleDimerTiler T(context, queue, devices, "./src/TriangleDimer/triangledimerkernel.cl", err);
    T.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", t.size());
    
    //Walk.
    std::random_device seed;
    T.Walk(t, Nsteps, seed());
    
    //Output final states.
    SaveMatrix(t,"./examples/TriangleDimer/DimerEnd.txt");
    TriangleDimerTiler::DimerToSVG(t,"./examples/TriangleDimer/DimerEnd.svg");
    TriangleDimerTiler::TilingToSVG(t,"./examples/TriangleDimer/TilingEnd.svg");
    
    
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(diff);
    std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
}

void RectTriangleEx(int N, int Nsteps) {
    // Rectangle-Triangle tilings of a hexagon with side length N.
    std::cout<<"Running Rectangle-Triangle Tiling Test Example."<<std::endl;
    auto start = std::chrono::steady_clock::now();
    
    //Standard OpenCL set up code.
    //PrintOpenCLInfo(); // Look at what devices are available
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    std::string sinfo;
    std::vector<cl::Device> devices;
    context.getInfo(CL_CONTEXT_DEVICES, &devices);
    devices[0].getInfo(CL_DEVICE_NAME, &sinfo);
    cl::CommandQueue queue(context);
    cl_int err = 0;
    
    //Create starting tilestate
    tiling tMax = RectTriangleTiler::maxHex(N);
    tiling tMin = RectTriangleTiler::minHex(N);
    tiling tSlope = RectTriangleTiler::slopeHex(N);
    SaveMatrix(tMax,"./examples/RectTriangle/MaxTiling.txt");
    SaveMatrix(tMin,"./examples/RectTriangle/MinTiling.txt");
    SaveMatrix(tSlope,"./examples/RectTriangle/SlopeTiling.txt");
    RectTriangleTiler::TilingToSVG(tMax,"./examples/RectTriangle/MaxTiling.svg");
    RectTriangleTiler::TilingToSVG(tMin,"./examples/RectTriangle/MinTiling.svg");
    RectTriangleTiler::TilingToSVG(tSlope,"./examples/RectTriangle/SlopeTiling.svg");
    
    //Set Up mcmc
    RectTriangleTiler R(context, queue, devices, "./src/RectTriangle/recttrianglekernel.cl", err);
    R.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", tMin.size());
    std::random_device seed;
    
    //Walk
    R.Walk(tMin, Nsteps, seed());
    
    //Output final state
    SaveMatrix(tMin,"./examples/RectTriangle/EndTiling.txt");
    RectTriangleTiler::TilingToSVG(tMin,"./examples/RectTriangle/EndTiling.svg");
    
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(diff);
    std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
    
}



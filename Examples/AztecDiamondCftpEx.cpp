/*
 * minimaldomino_example.cpp
 * 
 *
 * Tiles an Aztec Diamond of order N, using coupling from the past.
 */

#include "../src/common/common.h"
#include "../src/Domino/DominoTiler.h"

int main() {
	
	int N = 5;
	std::cout<<"Running Aztec Diamond of order "<<N<<" with CFTP."<<std::endl;

	//Standard OpenCL set up code.
	cl::Context context(CL_DEVICE_TYPE_DEFAULT);
	std::string sinfo;
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	devices[0].getInfo(CL_DEVICE_NAME, &sinfo); // Check which GPU you use!
	std::cout<<"Created context using: "<<sinfo<<std::endl;
	cl::CommandQueue queue(context);

	cl_int err = 0;

	
	// Make the domain. 
	domain d = DominoTiler::AztecDiamond(N);
	
	// Print the domain so you can see what it looks like:
	PrintMatrix(d);
	
	// And tile with the maximal and minimal height functions.
	tiling tMax = DominoTiler::MaxTiling(d);
	tiling tMin = DominoTiler::MinTiling(d);

	DominoTiler D(context, queue, devices, "./src/Domino/dominokernelCFTP.cl", err); // use the CFTP kernel!
	D.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", tMax.size()/2);

	// Seed the PRNG
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
			DominoTiler::TilingToSVG(top, "./Examples/ExampleOuts/AztecDiamondCFTP/RandomTiling.svg");
			break;
		}

		steps.insert(steps.begin(),steps[steps.size()-1]*2); seeds.insert(seeds.begin(),mt_rand());
	}
	
	std::cout<<"Finished!"<<std::endl;
}


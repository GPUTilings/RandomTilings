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
#include "./Lozenge/LozengeTiler.h"

int main() {
	std::cout<<"Running Lozenge, Basic Example."<<std::endl;
	auto start = std::chrono::steady_clock::now();
	int N = 10;
	int M = 2; 
	int Nsteps = 100;

	cl::Context context(CL_DEVICE_TYPE_DEFAULT);
	std::string sinfo;
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	devices[0].getInfo(CL_DEVICE_NAME, &sinfo);
	std::cout<<"Created context using: "<<sinfo<<std::endl;
	cl::CommandQueue queue(context);
	cl_int err = 0;

	//Creating and outputting starting tilestates. Here we use a hexagon domain minus a chunk, any domain will do though.
	domain d = LozengeTiler::AlmostHexDomain(N,M);
	tiling t = LozengeTiler::MaxTiling(d);
	LozengeTiler::DomainToSVG(d,"./ExampleTilings/MinimalLozenge/domainAlmostHex.svg");
	LozengeTiler::TilingToSVG(t, "./ExampleTilings/MinimalLozenge/tilingStartAlmostHex.svg");
	LozengeTiler::PrintDomain(d,"./ExampleTilings/MinimalLozenge/domainStartAlmostHex.txt");

	//Set up MCMC
	LozengeTiler L(context, queue, devices, "./src/lozenge/lozengekernel.cl", err);
	L.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", t.size());
	std::random_device seed;

	//Walk
	L.Walk(t, Nsteps, seed());

	//Output final states as tiling and dimer cover
	LozengeTiler::TilingToSVG(t, "./ExampleTilings/MinimalLozenge/tilingEndAlmostHex.svg");
	LozengeTiler::DimerToSVG(t, "./ExampleTilings/MinimalLozenge/dimerEndAlmostHex.svg");

	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	typedef std::chrono::duration<float> float_seconds;
	auto secs = std::chrono::duration_cast<float_seconds>(diff);
	std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
}


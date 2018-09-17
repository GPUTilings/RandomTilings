/*
 *  In this example, we will tile a hexagon of side length N with a corner of size M removed.
 *
 *
 */

#include "../src/common/common.h"
#include "../src/Lozenge/LozengeTiler.h"

int main() {
	std::cout<<"Running Lozenge, Basic Example."<<std::endl;
	auto start = std::chrono::steady_clock::now();
	
	
	int N = 10; int M = 2; 
	int Nsteps = 1000;

	cl::Context context(CL_DEVICE_TYPE_DEFAULT);
	std::string sinfo;
	std::vector<cl::Device> devices;
	context.getInfo(CL_CONTEXT_DEVICES, &devices);
	devices[0].getInfo(CL_DEVICE_NAME, &sinfo);
	std::cout<<"Created context using: "<<sinfo<<std::endl;
	cl::CommandQueue queue(context);
	cl_int err = 0;

	// Create a domain, and draw it out. See LozengeTiler.h for a description
	// of domains for lozenge tilings.
	domain d = LozengeTiler::AlmostHexDomain(N,M);
	LozengeTiler::DomainToSVG(d,"./Examples/ExampleOuts/MinimalLozenge/domainAlmostHex.svg");
	
	// Make an initial tiling.
	tiling t = LozengeTiler::MaxTiling(d);
	
	LozengeTiler::TilingToSVG(t, "./Examples/ExampleOuts/MinimalLozenge/tilingStartAlmostHex.svg");
	
	//Set up MCMC
	LozengeTiler L(context, queue, devices, "./src/lozenge/lozengekernel.cl", err);
	L.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", t.size());
	std::random_device seed;

	//Walk
	L.Walk(t, Nsteps, seed());

	//Output final states as tiling and dimer cover
	LozengeTiler::TilingToSVG(t, "./Examples/ExampleOuts/MinimalLozenge/tilingEndAlmostHex.svg");
	LozengeTiler::DimerToSVG(t, "./Examples/ExampleOuts/MinimalLozenge/dimerEndAlmostHex.svg");

	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	typedef std::chrono::duration<float> float_seconds;
	auto secs = std::chrono::duration_cast<float_seconds>(diff);
	std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
}


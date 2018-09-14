/*
 * minimaldomino_example.cpp
 * 
 *
 *  Created on: Dec 16, 2017
 *      Author: David, Ananth
 *
 * Tiles an equilateral triangle of side length N, oriented point downward, with a equillateral triangle of size M cut out of the top left corner.  
 */

#include "./common/common.h"
#include "./TriangleDimer/TriangleDimerTiler.h"

int main() {

	int N = 5;
	int M = 2;
	int Nsteps = 100;
	
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
    SaveMatrix(d,"./ExampleTilings/TriangleDimer/Domain.txt");
    SaveMatrix(t,"./ExampleTilings/TriangleDimer/DimerStart.txt");
    TriangleDimerTiler::DomainToSVG(d,"./ExampleTilings/TriangleDimer/Domain.svg");
    TriangleDimerTiler::DimerToSVG(t,"./ExampleTilings/TriangleDimer/DimerStart.svg");
    TriangleDimerTiler::TilingToSVG(t,"./ExampleTilings/TriangleDimer/TilingStart.svg");
    
    
    //Set up the mcmc.
    TriangleDimerTiler T(context, queue, devices, "./src/TriangleDimer/triangledimerkernel.cl", err);
    T.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", t.size());
    
    //Walk.
    std::random_device seed;
    T.Walk(t, Nsteps, seed());
    
    //Output final states.
    SaveMatrix(t,"./ExampleTilings/TriangleDimer/DimerEnd.txt");
    TriangleDimerTiler::DimerToSVG(t,"./ExampleTilings/TriangleDimer/DimerEnd.svg");
    TriangleDimerTiler::TilingToSVG(t,"./ExampleTilings/TriangleDimer/TilingEnd.svg");
    
    
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(diff);
    std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
}


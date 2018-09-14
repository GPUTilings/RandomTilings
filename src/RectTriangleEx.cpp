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
#include "./RectTriangle/RectTriangleTiler.h"

int main() {
	
	int N = 20; int Nsteps = 100;
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
    SaveMatrix(tMax,"./ExampleTilings/RectTriangle/MaxTiling.txt");
    SaveMatrix(tMin,"./ExampleTilings/RectTriangle/MinTiling.txt");
    SaveMatrix(tSlope,"./ExampleTilings/RectTriangle/SlopeTiling.txt");
    RectTriangleTiler::TilingToSVG(tMax,"./ExampleTilings/RectTriangle/MaxTiling.svg");
    RectTriangleTiler::TilingToSVG(tMin,"./ExampleTilings/RectTriangle/MinTiling.svg");
    RectTriangleTiler::TilingToSVG(tSlope,"./ExampleTilings/RectTriangle/SlopeTiling.svg");
    
    //Set Up mcmc
    RectTriangleTiler R(context, queue, devices, "./src/RectTriangle/recttrianglekernel.cl", err);
    R.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", tMin.size());
    std::random_device seed;
    
    //Walk
    R.Walk(tMin, Nsteps, seed());
    
    //Output final state
    SaveMatrix(tMin,"./ExampleTilings/RectTriangle/EndTiling.txt");
    RectTriangleTiler::TilingToSVG(tMin,"./ExampleTilings/RectTriangle/EndTiling.svg");
    
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(diff);
    std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
}


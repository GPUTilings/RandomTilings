//
//  Ex.cpp
//  
//
//  Created by david keating on 1/12/18.
//

#include <stdio.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <string>
#include <climits>
#include <sstream>
#include "./common/err_code.h"
#include <cmath>
#include <stack>
#include <random>
#include <chrono>
#include "./lozenge/LozengeTiler.h"


void LozengeBasicEx(int N, int M, int Nsteps) {
    try {
      
        
    } catch (cl::Error &err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
}

void LozengeqWeighted(int N, int M, int Nsteps) {
    try {
        std::cout<<"Running Lozenge, Basic Example."<<std::endl;
        auto start = std::chrono::steady_clock::now();
        
        //Standard OpenCL set up code.
        //PrintOpenCLInfo();
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
        tiling tiling = LozengeTiler::MaxTiling(d);
        LozengeTiler::DomainToSVG(d,"./examples/qWeightedHex/domain_qHex.svg");
        LozengeTiler::TilingToSVG(tiling, "./examples/qWeightedHex/tiling_qHex.svg");
        LozengeTiler::PrintDomain(d,"./examples/qWeightedHex/domain_qHex.txt");
        
        //Set up MCMC
        LozengeTiler L(context, queue, devices, "./src/lozenge/qlozengekernel.cl", err);
        L.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", tiling.size());
        std::random_device seed;
        
        //Walk
        std::cout<<"Starting a random walk of "<<Nsteps<<" moves...."<<std::endl;
        L.Walk(tiling, Nsteps, seed());
        //        L.Walk(tiling, Nsteps, 23849803);
        std::cout<<"done!"<<std::endl;
        
        //Output final states as tiling and dimer cover
        LozengeTiler::TilingToSVG(tiling, "./examples/qWeightedHex/tiling_qHex.svg");
        LozengeTiler::DimerToSVG(tiling, "./examples/qWeightedHex/dimer_qHex.svg");
        PrintMatrix(tiling,"./examples/qWeightedHex/tiling_EndqHex.txt");
        
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        typedef std::chrono::duration<float> float_seconds;
        auto secs = std::chrono::duration_cast<float_seconds>(diff);
        std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
        
    } catch (cl::Error &err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
}

void LozengeCFTP(int N, int M,  int Nsteps) {
    try {
        std::cout<<"Running Lozenge, Coupling from the Past."<<std::endl;
        auto start = std::chrono::steady_clock::now();
        
        //Standard OpenCL set up code.
        //Use PrintOpenCLInfo to see available devices.
        //PrintOpenCLInfo();
        cl::Context context(CL_DEVICE_TYPE_DEFAULT);
        std::string sinfo;
        std::vector<cl::Device> devices;
        context.getInfo(CL_CONTEXT_DEVICES, &devices);
        devices[0].getInfo(CL_DEVICE_NAME, &sinfo); //choose device
        cl::CommandQueue queue(context);
        cl_int err = 0;
        
        //Create starting tilings
        domain d = LozengeTiler::AlmostHexDomain(N,M);
        tiling maxTiling = LozengeTiler::MaxTiling(d);
        tiling minTiling = LozengeTiler::MinTiling(d);
        LozengeTiler::DomainToSVG(d,"./examples/HexCFTP/HexCFTPdomain.svg");
        LozengeTiler::TilingToSVG(maxTiling, "./examples/hexCFTP/HexCFTPstartT.svg");
        LozengeTiler::TilingToSVG(minTiling, "./examples/hexCFTP/HexCFTPstartB.svg");
        PrintMatrix(maxTiling,"./examples/hexCFTP/HexCFTPstartT.txt");
        
        
        //Set up MCMC
        LozengeTiler L(context, queue, devices, "./src/lozenge/lozengekernel.cl", err);
        L.LoadTinyMT("./src/TinyMT/tinymt32dc.0.1048576.txt", maxTiling.size());
        std::random_device seed;
        
        
        //Start CFPT
        std::stack<int> CFTP;
        
        std::default_random_engine generator;
        std::uniform_int_distribution<int> rands(0,INT_MAX);
        generator.seed(seed());
        //        generator.seed(5413234);
        
        tiling top;
        tiling bottom;
        
        while ( true ) {
            
            // Reset to start from the max and min tilings.
            top = maxTiling;
            bottom = minTiling;
            
            std::stack<int> randseeds(CFTP);
            
            int steps = 1;
            
            while (!randseeds.empty()) {
                L.Walk(bottom, steps*Nsteps, randseeds.top());
                L.Walk(top, steps*Nsteps, randseeds.top());
                randseeds.pop();
                
                steps *= 2;
            }
            
            heightfunc hbottom = LozengeTiler::TilingToHeightfunc(bottom,d);
            heightfunc htop = LozengeTiler::TilingToHeightfunc(top,d);
            
            int diff = 0;
            for (int i = 0; i < htop.size(); i++)
                diff += (htop[i] - hbottom[i]);
            
            if (diff == 0) {
                std::cout<<"Coupled!!"<<std::endl;
                break;
            }
            
            std::cout<<"Height difference: "<<diff<<std::endl;
            CFTP.push(rands(generator));
        }
        
        //Output final states
        LozengeTiler::TilingToSVG(top, "./examples/hexCFTP/HexCFTPendT.svg");
        LozengeTiler::TilingToSVG(bottom, "./examples/hexCFTP/HexCFTPendB.svg");
        LozengeTiler::DimerToSVG(top, "./examples/hexCFTP/HexCFTPendDimer.svg");
        
        auto end = std::chrono::steady_clock::now();
        auto diff = end - start;
        typedef std::chrono::duration<float> float_seconds;
        auto secs = std::chrono::duration_cast<float_seconds>(diff);
        std::cout<<"Size: "<<N<<".  Time elapsed: "<<secs.count()<<" s."<<std::endl;
        
    } catch (cl::Error &err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
    }
}

int main() {
    //LozengeBasicEx(20,8,10000);
    LozengeqWeighted(40,0,500000);
    //LozengeCFTP(4,0,1000);
    return 0;
}



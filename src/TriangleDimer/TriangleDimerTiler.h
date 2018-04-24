//
//  TriangleDimerTiler.h
//
//
//  Created by Ananth, David
//

#ifndef TRIANGLEDIMER_TRIANGLEDIMERTILER_H_
#define TRIANGLEDIMER_TRIANGLEDIMERTILER_H_

#include "../common/common.h"

class TriangleDimerTiler {

private:
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::make_kernel<cl::Buffer, const int> InitTinyMT;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int, const int> RotateLozenges;
    cl::make_kernel<cl::Buffer, const int, const int, const int> UpdateLozengesFlipped;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozenges0;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozenges1;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozenges2;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateTriangleUFromLozenges;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateButterflysHFromLozenge;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateButterflysLFromLozenge;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateButterflysRFromLozenge;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int> RotateTriangles;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateTrianglesFlipped0;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateTrianglesFlipped1;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int> UpdateTriangles;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozengeHFromTriangles;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozengeLFromTriangles;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozengeRFromTriangles;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int, const int, const int> RotateButterflys;
    cl::make_kernel<cl::Buffer, const int, const int, const int> UpdateButterflysFlippedH1;
    cl::make_kernel<cl::Buffer, const int, const int, const int> UpdateButterflysFlippedL1;
    cl::make_kernel<cl::Buffer, const int, const int, const int> UpdateButterflysFlippedR1;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedH21;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedH22;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedH23;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedL21;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedL22;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedL23;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedR21;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedR22;
    cl::make_kernel<cl::Buffer, const int, const int> UpdateButterflysFlippedR23;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozengeFromButterflysH;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozengeFromButterflysL;
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, const int> UpdateLozengeFromButterflysR;
	cl::Buffer tinymtparams;

public:
    
    // The constructor takes care of loading and compiling the program source.
    TriangleDimerTiler(cl::Context context0, cl::CommandQueue queue0, std::vector<cl::Device> devices, std::string source, cl_int &err) :
    	context(context0),
		queue(queue0),
    	program(LoadCLProgram(context,devices,source)),
    
        // Initialize TinyMT
		InitTinyMT(program, "InitTinyMT"),
    
        // Lozenge-type Flips
		RotateLozenges(program, "RotateLozenges"),
    
        // Kernels to update lozenges after flips. Number indicates orientation.
        UpdateLozengesFlipped(program, "UpdateLozengesFlipped"),
        UpdateLozenges0(program, "UpdateLozenges0"),
        UpdateLozenges1(program, "UpdateLozenges1"),
        UpdateLozenges2(program, "UpdateLozenges2"),
    
        // Kernels to update 'triangles' from lonzenges
        UpdateTriangleUFromLozenges(program, "UpdateTriangleUFromLozenges"),
    
        // Kernels to update 'butterflys' from lozenges
        UpdateButterflysHFromLozenge(program,"UpdateButterflysHFromLozenge"),
        UpdateButterflysLFromLozenge(program,"UpdateButterflysLFromLozenge"),
        UpdateButterflysRFromLozenge(program,"UpdateButterflysRFromLozenge"),
    
        // Triangle-type flips
        RotateTriangles(program,"RotateTriangles"),
    
        // Kernels to update 'traingles' after they flip. Number indicates orientation.
        UpdateTrianglesFlipped0(program,"UpdateTrianglesFlipped0"),
        UpdateTrianglesFlipped1(program,"UpdateTrianglesFlipped1"),
        UpdateTriangles(program,"UpdateTriangles"),
    
        // Kernels to update lozenges from 'triangles'
        UpdateLozengeHFromTriangles(program,"UpdateLozengeHFromTriangles"),
        UpdateLozengeLFromTriangles(program,"UpdateLozengeLFromTriangles"),
        UpdateLozengeRFromTriangles(program,"UpdateLozengeRFromTriangles"),
    
        // Butterfly-type flips
        RotateButterflys(program,"RotateButterflys"),
    
        // Kernels for updating 'butterflys' after they flip
        UpdateButterflysFlippedH1(program,"UpdateButterflysFlippedH1"),
        UpdateButterflysFlippedL1(program,"UpdateButterflysFlippedL1"),
        UpdateButterflysFlippedR1(program,"UpdateButterflysFlippedR1"),
        UpdateButterflysFlippedH21(program,"UpdateButterflysFlippedH21"),
        UpdateButterflysFlippedH22(program,"UpdateButterflysFlippedH22"),
        UpdateButterflysFlippedH23(program,"UpdateButterflysFlippedH23"),
        UpdateButterflysFlippedL21(program,"UpdateButterflysFlippedL21"),
        UpdateButterflysFlippedL22(program,"UpdateButterflysFlippedL22"),
        UpdateButterflysFlippedL23(program,"UpdateButterflysFlippedL23"),
        UpdateButterflysFlippedR21(program,"UpdateButterflysFlippedR21"),
        UpdateButterflysFlippedR22(program,"UpdateButterflysFlippedR22"),
        UpdateButterflysFlippedR23(program,"UpdateButterflysFlippedR23"),
    
        // Kernels for updating lozenge from butterfly
        UpdateLozengeFromButterflysH(program,"UpdateLozengeFromButterflysH"),
        UpdateLozengeFromButterflysL(program,"UpdateLozengeFromButterflysL"),
        UpdateLozengeFromButterflysR(program,"UpdateLozengeFromButterflysR") { };

    // TinyMT
    void LoadTinyMT(std::string params, int size);
    
    // Random Walk
    void Walk(tiling &t, int steps, long seed);

    // Create starting tilings
    static tiling IceCreamCone(int N, int M);
    static tiling TestButterfly(int t);
    
    // Methods for deomposing tiling into the different flip types
    static std::vector<int> TilingToTriangleFlips(tiling &t);
    static std::vector<int> TilingToButterflyFlips(tiling &t);
    static std::vector<int> TilingToVertices(tiling &t);
    
    // Method to get dimer configuration from tiling
    static std::vector<int> DimerIndicator(tiling &t, int c);
    
    // Get the domain from a tiling
    static domain TilingToDomain(tiling &t);
    
    // Methods for drawing to SVGs
    static void TilingToSVG(tiling &t, std::string filename);
    static void DimerToSVG(tiling &t, std::string filename);
    static void DimerToSVG(tiling &t, domain &d, std::string filename);
    static void DomainToSVG(domain &d, std::string filename);
};

#endif /* TRIANGLEDIMER_TRIANGLEDIMERTILER_H_ */

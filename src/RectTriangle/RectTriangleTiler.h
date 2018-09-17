//
//  RectTriangleTiler.h
//  
//
//  Created by Ananth, David
//



#ifndef RectTriangle_RectTriangleTiler_h
#define RectTriangle_RectTriangleTiler_h

#include "../common/common.h"


// Tiling is stored on vertices of the triangular lattice. Each adjacent face if given a integer value in [0,15] based on how it is covered by the tiling. These values are stored as a six digit hexidecimal int as follows:
//     ___
//   /\   /\         d4
//  /__\./__\ -> d5      d3 ->  d5*16^5 + d4*16^4 + d3*16^3 + d2*16^2 + d1*16^1 + d0*16^0
//  \  / \  /    d2      d0
//   \/___\/         d1
//
// The tiling is then tricolored such that all vertices of a given color can 'flip' without effecting each other. At each step in the Markov chain, it will first select a color then the gpu will attempt to flip all vertices of that color, then a seperate kernels will update the other colors based on the result of the flips (see recttrianglekernel.cl).
//
// Some technical notes:
// In order for dividing the tiling to work correctly it is neccessary that tiling is a square array with size divisible by 3. That is the size of tiling must be N*N, with N divisible by 3.
// In order for the update kernel to work correctly, the vector for each color must be padded by zeros; that is, the first and last rows and first and last columns must be zero. This is accomplished by making sure the tiling has 3 times the padding (first three and last three rows/cols are zero).


class RectTriangleTiler {
    
private:
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::make_kernel<cl::Buffer, const int> InitTinyMT;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int> flipTiles;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int> updateTiles1;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int> updateTiles2;
    cl::Buffer tinymtparams;
    
public:
    
    // The constructor takes care of loading and compiling the program source.
    RectTriangleTiler(cl::Context context0, cl::CommandQueue queue0, std::vector<cl::Device> devices, std::string source, cl_int &err) :
    context(context0),
    queue(queue0),
    program(LoadCLProgram(context,devices,source)),
    InitTinyMT(program, "InitTinyMT"),
    flipTiles(program, "flipTiles"),
    updateTiles1(program, "updateTiles1"),
    updateTiles2(program, "updateTiles2") { };
    
    // Load TinyMT
    void LoadTinyMT(std::string params, int size);
    
    // Random Walk
    void Walk(tiling &t, int steps, long seed);
    
    // Create Starting Tilings
    static tiling slopeHex(int N);
    static tiling maxHex(int N);
    static tiling minHex(int N);
    
    // Convert from Lozenge Tiling to RectTriangle Tiling
    static tiling LozengeToRectTriangle(tiling &tL, domain &dL);
    
    // Methods for drawing SVGs
    static void TilingToSVG(tiling &t, std::string filename);
    static std::string TilePicture(int TriType, int i, int j);
    
    
};

#endif /* RectTriangleTiler_h */

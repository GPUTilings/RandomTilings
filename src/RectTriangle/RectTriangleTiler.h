//
//  RectTriangleTiler.h
//  
//
//  Created by Ananth, David
//

#ifndef RectTriangle_RectTriangleTiler_h
#define RectTriangle_RectTriangleTiler_h

#include "../common/common.h"

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

//
//  TriangleDimerTiler.h
//
//
//  Created by Ananth, David
//

#ifndef TRIANGLEDIMER_TRIANGLEDIMERTILER_H_
#define TRIANGLEDIMER_TRIANGLEDIMERTILER_H_

#include "../common/common.h"

// There are three types of moves for our MCMC: Lozenge, Triangle, and Butterfly (see Perfect matchings in the triangular lattice - Kenyon, Remila). For each type of move, we have an array which, in each element, stores enough information about local state of the tiling to determine whether or not such a move can be executed. This is described below:
//
// Lozenge types moves:
//
// H: /\  L: __   R: __
//    \/    /__/    \__\
// with corresponding value on the edges:
//   1 2      1       1
//   8 4    8   2   8   2
//            4       4
//
// Each element of the tile state is stored on a vertex of the medial graph of the triangular lattice (extended to be another triangular lattice). In other words, at each edge of triangular lattice store the states of of the 4 edges on the two adjacent faces. Store 0 at each vertex of the triangular lattice between two horizontal edges for indexing purposes. For example for each edge going up and right there will be an associated state R as given above.
// Each type gets colored black and white in a checkerboard pattern, so that all black lozenges (of a given kind) can be flipped independently, similarly for white. Each call to the kernels we attempt to rotate all elements of a specific kind and color. Then  call a kernel to update all elements of same kind but different color. Then update the rest of the lozenges. Then update arrays of of Triangles and Butterflys.
//
// Triangle type moves:
//   ___   ___
//  \ 1 / \ 4 /        /16\
//   \ /___\ /   ,    / ___\
//    \ 16 /         / \   / \
//     \  /         /_1_\ /_4_\
//
//  We think of the center triangle holding the state, and the numbers in the adjacent triangle telling what to multiply the value of that triangle by when writing down the state (or what digit base 4 it gives). The value of each adjacent face given by:
//   /\
//  /__\ : 0 if no dimer, 1 if dimer on bottom edge, 2 if dimer on left edge, 3 if dimer on right edge
//   __
//  \  /
//   \/  : 0 if no dimer, 1 if dimer on top edge, 2 if dimer on left edge, 3 if dimer on right edge
//
// For the Triangle type flips, we construct a new array taking values on the faces of the triangular lattice. For each face the adjacent faces are given a value 0,1,2,3. These values are put together into a three digit number base 4. See diagram above.
// We divide the array into by 'up' and 'down' triangles (left and right picture above, respectively). These are then each tricolored in such a way that all 'up' (or down) elements of a given color can be flipped independently. A kernel first attempts to rotate all triangles of a given kind and color. Then update all triangle of same kind but different color. Then update rest of triangles. Then update Lozenges and Butterflys (update lozenge from triangle, then butterfly from lozenge).
//
// Butterfly type moves:
//
//     ___   ___              /1\            / 1\
// H: \ 1 / \ 4 /     L: ___ /___\       R: /___ \ ___
//     \ /___\ /   ,     \64/\   /\   ,    / \  / \ 4/
//     / \   / \          \/__\ /_4\      /_64\/___\/
//    /64_\ /16_\          \16/                \16 /
//                          \/                  \ /
// The state of a edge for a butterfly type move is stored as follows: each triangle around the center edge is given a weight as shown above. The weight of the edge is then w = e_1 + 4*e_4 + 16*e_16 + 64*e_64 + 256*c. The e_i are the value of the configuration of dimers around triangle_i. They are given by:
//
//          __2_       /\             __1_       /\            __3_       /\            __2_       /\
//  e_1:   1\  /3    1/__\2 ,   e_4: 3\  /2    3/__\1 , e_16: 2\  /1    3/__\1 , e_64: 1\  /3    2/__\3
//           \/         3              \/         2             \/         2             \/         1
//
//  where the number indicates the value e_i takes if a dimer occupies that edge. e_i is zero if no dimers occupy the edges.
//  c is 1 if the center edge of H,L, or R is covered by a dimer. It is zero otherwise.
//
//  For butterfly flips, we first construct an array of butterfly states from the array of lozenge states. We divide it into three arrays, one for each type of butterfly (H,L,R). When attempting a flip we first divide the array into slices mod 3 (rows mod 3 for H, colums mod 3 for L, lines i+j=const mod 3 for R) and pick one of the three colors of slice, we then choose to flip even or odd sites on the chosen slice.
//  After flips are attempted, update the type that was flipped, then use that to update the lozenges, then use the lozenges to update the rest of the butterflys and the triangles.
//
// Some technical notes:
// The tiling is stored as an array of lozenge states. The other types of states (Triangle and Butterfly) are create from the lozenge states.
// Due to how we divide the lozenge states, the size of the tiling must be square and divisible by 2. That is, the tiling must be N*N with N divisible by 2.
// Due to how the update kernels operate, the first three and the last three rows and columns of the tiling must be 0. This ensures the vectors of lozenge states are adequately padded, as well as ensuring that the Triangle and Butterfly tile vectors are also adequately padded.
// Icecreamcone creates a tiling that is of the correct size with the required padding.
// There are a million kernels and its a huge mess :/


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

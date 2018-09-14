#ifndef DOMINO_DOMINOTILER_H_
#define DOMINO_DOMINOTILER_H_

/*
 * dominoTiler.h
 *
 *  Created on: Dec 16, 2017
 *      Author: David, Ananth
 *
 * This is a class for doing Markov chain simulations of domino tilings
 * with the GPU, along with utility functions for computing height functions,
 * for computing maximal and minimal tilings, for constructing some standard domains,
 * and for drawing tilings to SVG files.
 *
 * See the domino the examples for usage examples.
 *
 * Some definitions for domino tilings:
 *
 * A domain (typedef of a std::vector<int>, see common.h) is an array of size NxN describing a
 * domain of the square lattice, with the (i*N+j) element equal to one if the square (i,j) of the square
 * lattice is in the domain.
 *
 * A tiling (typedef of a std::vector<int>, see common.h) is a square array representing a domino tiling.
 * The (i*N+j) element of the tiling is the state of the vertex (i,j) in the tiling.
 * The state is found by summing an indicator function on adjacent edges,
 * which is nonzero if a domino (or equivalently a dimer) crosses the edge and equal to:
 *
 *      1|
 * 8 ____|____ 4
 *       |
 *      2|
 *
 *
 * It is important that the tiling has dimension of the form 2Mx2M.
 *
 * It is important that the tiling is twice zero-padded, i.e. that all vertices
 * at the boundary of the array, and their neighbors be zero.
 */


#include "../common/common.h"

class DominoTiler {

private:
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::make_kernel<cl::Buffer, const int> InitTinyMT;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int> RotateTiles;
    cl::make_kernel<cl::Buffer, cl::Buffer, const int, const int> UpdateTiles;
	cl::Buffer tinymtparams;

public:

	/*
	 * here are the core methods
	 */

	// The constructor takes care of loading and compiling the program source.
    DominoTiler(cl::Context context0, cl::CommandQueue queue0, std::vector<cl::Device> devices, std::string source, cl_int &err) :
    	context(context0),
		queue(queue0),
    	program(LoadCLProgram(context,devices,source)),
		InitTinyMT(program, "InitTinyMT"),
		RotateTiles(program, "RotateTiles"),
		UpdateTiles(program, "UpdateTiles") { };

    // Loads TinyMT parameters.
    void LoadTinyMT(std::string params, int size);

    // Random Walk the tiling for STEPS steps with seed SEED.
    void Walk(tiling &t, long steps, long seed);

    // Random Walk the tiling with a vector of steps, and a vector of seeds.
    void Walk(tiling &t, std::vector<long> steps, std::vector<long> seeds);

    /*
     * The following functions are utility functions for constructing domains, tilings
     * and drawing.
     */

    // Returns an MxN rectangular domain.
    static domain Rectangle(int M, int N);

    // Returns an Aztec Rectangle domain with the points REMOVE removed. See "Random domino tilings of
    // Rectangular Aztec Diamonds" by Knizel, Bufetov,.
    static domain AztecRectangle(int N, const std::vector<int> &remove);

    // Aztec Rectangle
    static domain AztecDiamond(int N);

    // Implementation of Thurston's algorithm to construct maximal and minimal tilings.
    static tiling MaxTiling(const domain &d);
    static tiling MinTiling(const domain &d);

    // Given a domain, returns the vertices in the domain.
    // The vertices are represented by a square 0,1 array,
    // with the (i*N+j) element 1 iff the vertex is in the domain.
    static std::vector<int> DomainToVertices(const domain &d);

    // Given a tiling, returns the domain.
    static domain TilingToDomain(const tiling &t);

    // Convert between a tiling and a height function.
    static tiling HeightfuncToTiling(const heightfunc &hf, const domain &d);
    static heightfunc TilingToHeightfunc(const tiling &t, const domain &d);

    // A bunch of methods for drawing Scalable Vector Graphics.
    static void DomainToSVG(const domain &d, std::string filename); // Draws the domain.
    static void TilingToSVG(const tiling &t, std::string filename); // Draws the tiling.
    static void MayaToSVG(const tiling &t, std::string filename); // Draws the corresponding Maya diagrams.
    static void DimerToSVG(const tiling &t, std::string filename); // Draws the corresponding dimer model.
    static void DimerToSVG(const tiling &t, const domain &d, std::string filename); // Draws the corresponding dimer model forced on the domain d.
};

#endif /* DOMINO_DOMINOTILER_H_ */

# Random Tilings with the GPU
<img align="right" width="200" src="https://github.com/LittleBadger/RandomTilings/blob/master/TriangleTiling.svg">
Here is an C++/OpenCL library for generating random tilings efficiently with Markov Chain Monte Carlo on the GPU. See the companion paper [1] for further details.  At the moment, the library supports domino tilings, lozenge tilings, bibone tilings, rectangle-triangle tilings, and the six vertex model, and also includes utility functions for constructing domains, maximal/minimal tilings, height functions, Maya diagrams, lattice paths, etc. The program can outputs tilings as [Scalable Vector Graphics](https://en.wikipedia.org/wiki/Scalable_Vector_Graphics) or raw as a table suitable for analysis. 

## Prerequisites
In order to build and run the program, you will need the following:
* [OpenCL](http://www.khronos.org/opencl)
* [TinyMT](https://github.com/MersenneTwister-Lab/TinyMT): A library for generating random numbers on the GPU, see [here](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/).
* [TinyMT Parameters](https://github.com/jj1bdx/tinymtdc-longbatch): A list of precomputed parameters for the TinyMT number generator.

## Building And Running
The structure of the library is as follows: . Certain common functionality.

A sample makefile, and several examples are contained in the root folder. To run the example, run:
....



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details



## References
[1] D. Keating, A. Sridhar. "Random Tilings with the GPU." https://arxiv.org/pdf/1804.07250.pdf
[2] M. Fisher, B. Springborn, P. Schröder, A. I. Bobenko. "An algorithm for the construction of intrinsic delaunay triangulations with applications to digital geometry processing." ACM SIGGRAPH 2006 Courses, 69-74.\
[3] U. Pinkall, K. Polthier. Experiment. Math., Volume 2, Issue 1 (1993), 15-36.

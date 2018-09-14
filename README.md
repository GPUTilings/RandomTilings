# Random Tilings with the GPU
Here is an C++/OpenCL library for generating random tilings efficiently with Markov Chain Monte Carlo on the GPU. At the moment, the library supports domino tilings, lozenge tilings, bibone tilings, rectangle-triangle tilings, and the six vertex model. See the companion paper [1] for further details.

<img align="right" width="200" src="https://github.com/LittleBadger/RandomTilings/blob/master/TriangleTiling.svg"><img align="right" width="200" src="https://github.com/LittleBadger/RandomTilings/blob/master/DominoTiling.svg">


## Prerequisites
In order to run the program, you will need:
* [OpenCL](http://www.khronos.org/opencl)
* [TinyMT](https://github.com/MersenneTwister-Lab/TinyMT): A library for generating random numbers on the GPU.

## Building And Running
The 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details



## References
[1] D. Keating, A. Sridhar. "Random Tilings with the GPU." https://arxiv.org/pdf/1804.07250.pdf
[2] M. Fisher, B. Springborn, P. Schröder, A. I. Bobenko. "An algorithm for the construction of intrinsic delaunay triangulations with applications to digital geometry processing." ACM SIGGRAPH 2006 Courses, 69-74.\
[3] U. Pinkall, K. Polthier. Experiment. Math., Volume 2, Issue 1 (1993), 15-36.

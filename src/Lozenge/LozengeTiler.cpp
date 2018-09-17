#include "LozengeTiler.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <climits>
#include <queue>
#include <random>
#include "../common/common.h"
#include "../TinyMT/file_reader.h"


const int neighbors[6][2] = {{0,1},{-1,1},{-1,0},{0,-1},{1,-1},{1,0}};
const int dh[6] = {-1,1,-1,1,-1,1};
const int neighborfaces[6][2][2] = {{{0,0},{-1,1}}, {{-1,1},{-1,0}},{{-1,0},{-1,-1}},
		{{-1,-1},{0,-2}},{{0,-2},{0,-1}},{{0,-1},{0,0}}};
const int pws[6] = {8,2,1,4,16,32};

void LozengeTiler::Walk(tiling &t, int steps, long seed) {

	int N = sqrt(t.size());
    int M = N/3;
    
    // Iniitialize host vectors
    std::vector<char> h_vR(N*M,0);
    std::vector<char> h_vB(N*M,0);
    std::vector<char> h_vG(N*M,0);
    
    // Fill the host vectors
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            if (((j-i)%3+3)%3 == 0) {
                h_vR[i*M+j/3] = t[i*N+j];
            } else if (((j-i)%3+3)%3 == 1) {
                h_vB[i*M+j/3] = t[i*N+j];
            } else {
                h_vG[i*M+j/3] = t[i*N+j];
            }
        }
    }
 
    // To the buffer!
    cl::Buffer d_vR = cl::Buffer(context, h_vR.begin(), h_vR.end(), CL_FALSE);
    cl::Buffer d_vB = cl::Buffer(context, h_vB.begin(), h_vB.end(), CL_FALSE);
    cl::Buffer d_vG = cl::Buffer(context, h_vG.begin(), h_vG.end(), CL_FALSE);
    
    // TinyMT
	InitTinyMT( cl::EnqueueArgs(queue, cl::NDRange(N*(N/3))), tinymtparams, seed );
    
    // MCMC
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,2);
    generator.seed(seed);

	for(int i=0; i < steps; ++i) {
        int r = distribution(generator);
        //std::cout<<r<<std::endl;
        if (r == 0) {
            RotateTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), tinymtparams, d_vR, N);
            UpdateTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vR, d_vB, d_vG, N, 0);
        } else if (r == 1) {
            RotateTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), tinymtparams, d_vB, N);
            UpdateTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vB, d_vG, d_vR, N, 1);
        } else {
            RotateTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), tinymtparams, d_vG, N);
            UpdateTiles( cl::EnqueueArgs(queue, cl::NDRange(N,N/3)), d_vG, d_vR, d_vB, N, 2);
        }
		
	}
    
    // Copy back to the host
    cl::copy(queue, d_vR, h_vR.begin(), h_vR.end());
    cl::copy(queue, d_vB, h_vB.begin(), h_vB.end());
    cl::copy(queue, d_vG, h_vG.begin(), h_vG.end());
	
    // Recombine into tiling
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            if (((j-i)%3+3)%3 == 0) {
                t[i*N+j] = h_vR[i*M+j/3];
            } else if (((j-i)%3+3)%3 == 1) {
                t[i*N+j] = h_vB[i*M+j/3];
            } else {
                t[i*N+j] = h_vG[i*M+j/3];
            }
        }
    }

}


void LozengeTiler::LoadTinyMT(std::string params, int size) {
	tinymtparams = get_params_buffer(params, context, queue, size);
}

heightfunc LozengeTiler::TilingToHeightfunc(const tiling &t, const domain &d) {
	int N = sqrt(t.size());
	int M = N-1;
    

	std::vector<int> vertices = DomainToVertices(d);
	std::vector<int> hf(vertices.size(),infty);

	std::queue<std::pair<int, int> > bfs;

	bool found = false;

	for (int i = 0; i < N && !found; i++) {
		for (int j = 0; j < N && !found; j++) {   // Search through each point in the domain.
			if ( vertices[i*N+j] == 2 ) {   // Check if it is a boundary point.
				found = true;   // If we found a boundary point stop searching,
				bfs.push(std::pair<int, int>(i,j));   // add it to the queue,
				hf[i*N+j] = 0;   // and set its height to 0.
			}
		}
	}

	while ( !bfs.empty() )
	{
		int i0 = bfs.front().first; int j0 = bfs.front().second;
		bfs.pop();

		for (int nb = 0; nb < 6; nb++) {
			if ( (t[i0*N+j0] & pws[nb]) == 0 ) {
				int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];

				// if the neighbor is in the domain, is unvisited, and the path to the neighbor is actually in the domain...
				if ( vertices[i*N+j] > 0 && hf[i*N+j] == infty &&
						(d[(i0+neighborfaces[nb][0][0])*2*M + 2*j0 + neighborfaces[nb][0][1] ] || d[(i0+neighborfaces[nb][1][0])*2*M + 2*j0 + neighborfaces[nb][1][1] ]) ) {
					hf[i*N+j] = hf[i0*N+j0] + dh[nb];
					std::pair<int,int> np = std::pair<int, int>(i,j);
					bfs.push(np);
				}
			}
		}
	}

	return hf;
}

tiling LozengeTiler::MinTiling(const domain &d) {
	std::vector<int> vertices = DomainToVertices(d);

	int M = sqrt(d.size()/2);
	int N = sqrt(vertices.size());

	std::vector<int> hf(N*N,infty); // creates a vector of ints called hf of size N*N with all elements having value infty

	/*
	 * The height function rules on the triangular lattice:
	 * Each vertex is 6-valent. Starting from the Eastern edge, the height changes are:
	 * -1, +1, -1, +1, -1, +1
	 */

	// We assume the domain is connected and simply connected.

	// First we find a starting vertex on the boundary of the domain
	// and at it to the BFS queue that we will use to compute the boundary heights.
	std::queue<std::pair<int, int> > bfs;

	bool found = false;

	for (int i = 0; i < N && !found; i++) {
		for (int j = 0; j < N && !found; j++) {   // Search through each point in the domain.
			if ( vertices[i*N+j] == 2 ) {   // Check if it is a boundary point.
				found = true;   // If we found a boundary point stop searching,
				bfs.push(std::pair<int, int>(i,j));   // add it to the queue,
				hf[i*N+j] = 0;   // and set its height to 0.
			}
		}
	}

	// Priority queue for Dijkstra purpose
	std::priority_queue< std::pair<std::pair<int, int>,int>, std::vector<std::pair<std::pair<int, int>, int> >, incheightPriority > bfs_pq( ( incheightPriority() ));   // Defines bfs_pq. This contains the boundary points ordered from least height to greatest. So we'll pop the lowest height first.

	// boundary stuff here
	while ( !bfs.empty() )
	{
		int i0 = bfs.front().first; int j0 = bfs.front().second;
		bfs.pop();

		for (int nb = 0; nb < 6; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];
			if ( vertices[i*N+j] == 2) {
				if ( d[(i0+neighborfaces[nb][0][0])*2*M + 2*j0 + neighborfaces[nb][0][1] ] ^ d[(i0+neighborfaces[nb][1][0])*2*M + 2*j0 + neighborfaces[nb][1][1] ]) { // check that we are traversing a boundary edge
					if ( hf[i*N+j] == infty ) {
						std::pair<int,int> np = std::pair<int, int>(i,j);
						hf[i*N+j] = hf[i0*N+j0] + dh[nb];
						bfs.push(np);
						bfs_pq.push(std::pair<std::pair<int,int>, int>(np, hf[i*N+j]));
					} else {
						if (hf[i*N+j] != hf[i0*N+j0] + dh[nb] ) {
							std::cout<<"!!This region is NOT tileable!! Min, Boundary"<<std::endl;
							return HeightfuncToTiling(hf, d);
						}
					}
				}
			}
		}
	}

	while (!bfs_pq.empty()) {
		int i0 = bfs_pq.top().first.first; int j0 = bfs_pq.top().first.second; bfs_pq.pop();

		for (int nb = 0; nb < 6; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];

			if ( vertices[i*N+j] > 0 && dh[nb] < 0) {   // Only move in directions that decrease height.

				if ( vertices[i*N+j] == 2 && hf[i*N+j] < hf[i0*N+j0] + dh[nb]) {   // Check height compatibility. Will the catch all cases?
					std::cout<<"!!This region is NOT tileable!! Min, Bulk"<<std::endl;  // what about checking with already visted site not on boundary?
					return HeightfuncToTiling(hf, d);
				} else if ( hf[i*N+j] == infty ) {
					hf[i*N+j] = hf[i0*N+j0] + dh[nb];
					bfs_pq.push(std::pair<std::pair<int,int>, int >(std::pair<int, int>(i,j),hf[i*N+j]));
				}
			}
		}
	}

	for (int i = 0; i < N; i++) {   // Set height of all vertices outside domain to 0.
		for (int j = 0; j < N; j++) {
			if ( vertices[i*N+j] == 0 )
				hf[i*N+j] = infty;
		}
	}

	return HeightfuncToTiling(hf, d);
}

tiling LozengeTiler::MaxTiling(const domain &d) {
	std::vector<int> vertices = DomainToVertices(d);

	int M = sqrt(d.size()/2);
	int N = sqrt(vertices.size());

	std::vector<int> hf(N*N,infty); // creates a vector of ints called hf of size N*N with all elements having value infty

	/*
	 * The height function rules on the triangular lattice:
	 * Each vertex is 6-valent. Starting from the Eastern edge, the height changes are:
	 * -1, +1, -1, +1, -1, +1
	 */

	// We assume the domain is connected and simply connected.

	// First we find a starting vertex on the boundary of the domain
	// and at it to the BFS queue that we will use to compute the boundary heights.
	std::queue<std::pair<int, int> > bfs;

	bool found = false;

	for (int i = 0; i < N && !found; i++) {
		for (int j = 0; j < N && !found; j++) {   // Search through each point in the domain.
			if ( vertices[i*N+j] == 2 ) {   // Check if it is a boundary point.
				found = true;   // If we found a boundary point stop searching,
				bfs.push(std::pair<int, int>(i,j));   // add it to the queue,
				hf[i*N+j] = 0;   // and set its height to 0.
			}
		}
	}

	// Priority queue for Dijkstra purpose
	std::priority_queue< std::pair<std::pair<int, int>,int>, std::vector<std::pair<std::pair<int, int>, int> >, decheightPriority > bfs_pq( ( decheightPriority() ));   // Defines bfs_pq. This contains the boundary points ordered from least height to greatest. So we'll pop the lowest height first.

	// boundary stuff here
	while ( !bfs.empty() )
	{
		int i0 = bfs.front().first; int j0 = bfs.front().second;
		bfs.pop();

		for (int nb = 0; nb < 6; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];
			if ( vertices[i*N+j] == 2) {
				if ( d[(i0+neighborfaces[nb][0][0])*2*M + 2*j0 + neighborfaces[nb][0][1] ] ^ d[(i0+neighborfaces[nb][1][0])*2*M + 2*j0 + neighborfaces[nb][1][1] ]) { // check that we are traversing a boundary edge
					if ( hf[i*N+j] == infty ) {
						std::pair<int,int> np = std::pair<int, int>(i,j);
						hf[i*N+j] = hf[i0*N+j0] + dh[nb];
						bfs.push(np);
						bfs_pq.push(std::pair<std::pair<int,int>, int>(np, hf[i*N+j]));
					} else {
						if (hf[i*N+j] != hf[i0*N+j0] + dh[nb] ) {
							std::cout<<"!!This region is NOT tileable!! Max, Boundary"<<std::endl;
							return HeightfuncToTiling(hf, d);
						}
					}
				}
			}
		}
	}

	// now fill in the height function in the bulk. Start at heighest vertex and only move down.
	while (!bfs_pq.empty()) {
		int i0 = bfs_pq.top().first.first; int j0 = bfs_pq.top().first.second; bfs_pq.pop();

		for (int nb = 0; nb < 6; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];

			if ( vertices[i*N+j] > 0 && dh[nb] > 0) {   // Only move in directions that decrease height.

				if ( vertices[i*N+j] == 2 && hf[i*N+j] > hf[i0*N+j0] + dh[nb]) {   // Check height compatibility. Will the catch all cases?
					std::cout<<"!!This region is NOT tileable!! Min, Bulk"<<std::endl;  // what about checking with already visted site not on boundary?
					return HeightfuncToTiling(hf, d);
				} else if ( hf[i*N+j] == infty ) {
					hf[i*N+j] = hf[i0*N+j0] + dh[nb];
					bfs_pq.push(std::pair<std::pair<int,int>, int >(std::pair<int, int>(i,j),hf[i*N+j]));
				}
			}
		}
	}

	for (int i = 0; i < N; i++) {   // Set height of all vertices outside domain to 0.
		for (int j = 0; j < N; j++) {
			if ( vertices[i*N+j] == 0 )
				hf[i*N+j] = infty;
		}
	}
	return HeightfuncToTiling(hf, d);
}


tiling LozengeTiler::HeightfuncToTiling(const heightfunc &hf, const domain &d) {
	int N = sqrt(hf.size());
	int M = N-1;

	std::vector<int> tiling(N*N,0);

	for(int i0=1; i0<N-1; ++i0) {
		for(int j0=1; j0<N-1; ++j0) {

			for (int nb = 0; nb < 6; nb++) {

				int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];
				if (d[(i0+neighborfaces[nb][0][0])*2*M + 2*j0 + neighborfaces[nb][0][1] ] || d[(i0+neighborfaces[nb][1][0])*2*M + 2*j0 + neighborfaces[nb][1][1] ] ) {
					if ((std::abs(hf[i*N+j] - hf[i0*N+j0])) == 2) {
						tiling[i0*N+j0] += pws[nb];
					}
				}
			}
		}
	}
	return tiling;
}

domain LozengeTiler::AlmostHexDomain(int N, int M) {
    domain d(2*N*4*N,0);
    domain d2((2*N+6)*(4*N+12),0);
    if(M>N) {
        std::cout<<"Cutting out to big a chunk."<<std::endl;
        return d2;
    }
    for(int i=0; i<2*N; ++i) {
        for(int j=0; j<4*N; ++j) {
            if(i < N) {
                if(2*i + j-1 >= 2*N-2 ) {
                    d[i*4*N+j] = 1;
                }
            } else {
                if(2*i + j <= 6*N-2) {
                    d[i*4*N+j] = 1;
                }
            }
        }
    }
    for(int i=0; i<M; ++i) {
        for(int j=4*N-2*M; j<4*N; ++j) {
            d[i*4*N+j] = 0;
        }
    }
    
    // add padding
    for(int i = 3; i<2*N+3; ++i) {
        for(int j = 6; j<4*N+6; ++j) {
            d2[i*(4*N+12)+j] = d[(i-3)*4*N+j-6];
        }
    }
    
    // make sure tiling will be divisible by 3
    int M2 = sqrt(d2.size()/2);
    int k = (3-(M2+1)%3)%3;
    domain d3((M2+k)*2*(M2+k),0);
    for(int i = 0; i<M2; ++i) {
        for(int j = 0; j<2*M2; ++j) {
            d3[i*2*(M2+k)+j] = d2[i*2*M2+j];
        }
    }
    
    
    return d3;
}

void LozengeTiler::PrintDomain(const domain &d, std::string filename) {
    std::ofstream outputFile(filename.c_str());
    
    int N = sqrt(d.size()/2);
    for(int i=0; i<N; ++i) {
        for(int j=0; j<2*N; ++j) {
            outputFile<<d[i*2*N+j]<< " ";
        }
        outputFile<<"\n";
    }
    
    outputFile.close();
}

std::vector<int> LozengeTiler::DomainToVertices(const domain &d) {
	int M = sqrt(d.size()/2);
	int N = M+1;

	std::vector<int> vertices(N*N,0);

	for (int i=0; i<N; ++i) {
		for (int j=0; j<N; ++j) {
			if (!(i == 0 || j == 0 || i == M || j == M)) {

				if (d[i*2*M + 2*j]
					  && d[i*2*M + 2*j-1]
						   && d[i*2*M + 2*j-2]
								&& d[(i-1)*2*M + 2*j]
									 && d[(i-1)*2*M + 2*j+1]
										  && d[(i-1)*2*M + 2*j-1] )
				{
					vertices[i*N+j] = 1;
				} else if (d[i*2*M + 2*j] ||
						d[i*2*M + 2*j-1] ||
						d[i*2*M + 2*j-2] ||
						d[(i-1)*2*M + 2*j] ||
						d[(i-1)*2*M + 2*j+1] ||
						d[(i-1)*2*M + 2*j-1])
				{vertices[i*N+j] = 2; }
			}
		}
	}
	return vertices;
}

domain LozengeTiler::TilingToDomain(const tiling &t) {
	int N = sqrt(t.size());
	int M = N-1;

	domain d((N-1)*(N-1)*2,0);

	for (int i = 0; i < N-1; i++) {
		for (int j = 0; j < N-1; j++) {

			if ( (t[i*N+j] & 1) != 0 ) {
				d[(i-1)*2*M + 2*j] = 1;
				d[(i-1)*2*M + 2*j-1] = 1;
			}

			if ( (t[i*N+j] & 2) != 0 ) {
				d[(i-1)*2*M + 2*j] = 1;
				d[(i-1)*2*M + 2*j+1] = 1;
			}

			if ( (t[i*N+j] & 8) != 0 ) {
				d[(i-1)*2*M + 2*j+1] = 1;
				d[(i)*2*M + 2*j] = 1;
			}
		}
	}
	return d;
}

/*
 * Here begin bunch of tools for drawing. Here are some useful geometric formulas:
 *
 * Vertex (i,j) of the equaliteral lattice has coordinate
 * x = i * 1/2.0 + j
 * y = i * sqrt3 / 2.0
 *
 * The center of the (i,j) triangle of the equilateral lattice has the coordinate:
 * x = .5 + i * .5 + j
 * y = i * sqrt3 / 2 + sqrt3 / 4 + (j%2) * (sqrt3 / 3 - sqrt3 / 4)
 *
 * The (i,j) face is adjacent to vertices:
 * if j is even: (i,j/2), (i+1,j/2), (i,j/2+1)
 * if j is odd: (i+1,j/2+1), (i+1,j/2), (i,j/2+1)
 */

void LozengeTiler::DimerToSVG(const tiling &t, std::string filename) {
	domain d = TilingToDomain(t);
	LozengeTiler::DimerToSVG(t,d,filename);
}


//layers of this function are wrong
void LozengeTiler::DimerToSVG(const tiling &t, const domain &d, std::string filename) {

	int N = sqrt(t.size());
	int M = N-1;

	//double W = sqrt3*2*(N+1); double H = 3/2.0*(N+1);

	std::ofstream outputFile(filename.c_str());

	outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"433.013\" width=\"500\" viewBox=\"0 0 "<<3/2.0*M<<" "<<3/2.0*M<<"\" preserveAspectRatio=\"none\">\n";

	outputFile <<"<defs>\n";
	outputFile << "<g id=\"e1\">  <polyline points= \"-.5,-.33333333333 0,-.666666666666\" stroke=\"black\" stroke-width=\".001\"/> </g>\n";
	outputFile << "<g id=\"e2\">  <polyline points= \"0,-.666666666 .5,-.333333333333\" stroke=\"black\" stroke-width=\".001\"/> </g>\n";
	outputFile << "<g id=\"e8\">  <polyline points= \".5,-.333333333333 .5,.333333333333\" stroke=\"black\" stroke-width=\".001\"/> </g>\n";

	outputFile << "<g id=\"d1\">  <polyline points= \"-.5,-.33333333333 0,-.666666666666\" stroke=\"black\" stroke-width=\".025\"/> </g>\n";
	outputFile << "<g id=\"d2\">  <polyline points= \"0,-.666666666 .5,-.333333333333\" stroke=\"black\" stroke-width=\".025\"/> </g>\n";
	outputFile << "<g id=\"d8\">  <polyline points= \".5,-.333333333333 .5,.333333333333\" stroke=\"black\" stroke-width=\".025\"/> </g>\n";

	outputFile << "<g id=\"b\">  <ellipse cx=\".5\" cy=\".333333333333\" rx=\".1\" ry=\".11547\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/> </g> \n";
	outputFile << "<g id=\"w\">  <ellipse cx=\".5\" cy=\".666666666666\" rx=\".1\"  ry=\".11547\" stroke=\"black\" stroke-width=\".02\" fill=\"white\" /> </g>\n";
	outputFile <<"</defs>\n";

	//draw dimers

	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < N-1; j++) {
			if ( (t[i*N+j] & 1) == 1 ) {
				outputFile<<"<use xlink:href = \"#d1\" x = \""<<(j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
			}
			if ( (t[i*N+j] & 2) == 2 ) {
				outputFile<<"<use xlink:href = \"#d2\" x = \""<<(j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
			}
			if ( (t[i*N+j] & 8) == 8 ) {
				outputFile<<"<use xlink:href = \"#d8\" x = \""<<(j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
			}
		}
	}

	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < 2*N-1; j++) {
			if (d[i*M*2+j] == 1) {
				// to do: one should draw the non-dimer edges
if ( j% 2 == 0) {
				if ( d[i*M*2 +j-1] == 1) {
					outputFile<<"<use xlink:href = \"#e1\" x = \""<<(.5+.5*j+.5*i)<<"\" y = \""<<i+1<<"\"/>\n"; }
				if ( d[i*M*2 +j+1] == 1) {
					outputFile<<"<use xlink:href = \"#e2\" x = \""<<(.5+.5*j+.5*i)<<"\" y = \""<<i+1<<"\"/>\n";}
				if ( d[(i+1)*M*2 +j-1] == 1 ) {
					outputFile<<"<use xlink:href = \"#e8\" x = \""<<(.5+.5*j+.5*i)<<"\" y = \""<<i+1<<"\"/>\n";
				}}


			}
		}
}

	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < 2*N-1; j++) {
			if (d[i*M*2+j] == 1) {
				// draw vertices:
				double x = .5*i+.5*j;
				double y = i;
					if ( j%2 == 0)
						outputFile<<"<use xlink:href = \"#b\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
					else
						outputFile<<"<use xlink:href = \"#w\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";

			}}
			}

	outputFile<< "</svg>";
	outputFile.close();
}



void LozengeTiler::TilingToSVG(const tiling &t, std::string filename) {
	std::ofstream outputFile(filename.c_str());
	int N = sqrt(t.size());
	int M = N-1;

	outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"433.013\" width=\"500\" viewBox=\"0 0 "<<3/2.0*M<<" "<<3/2.0*M<<"\" preserveAspectRatio=\"none\">\n";
	outputFile << "<defs>\n";
	outputFile << "<g id=\"r\">  <polygon points = \"0,0 -1,0 -.5,-1 .5,-1\" fill=\"midnightblue\"/> </g>\n";
	outputFile << "<g id=\"g\">  <polygon points = \"0,0 1,0 .5,-1 -.5,-1\" fill=\"lightsteelblue\"/> </g>\n";
	outputFile << "<g id=\"b\">  <polygon points = \"0,0 .5,-1 1,0 .5,1\" fill=\"slategrey\"/> </g>\n";
	outputFile<<"</defs>\n";

	//draw dimers

	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < N-1; j++) {
			if ( (t[i*N+j] & 1) == 1 ) { outputFile<<"<use xlink:href = \"#r\" x = \""<<(j+.5*i)<<"\" y = \""<<i<<"\"/>\n"; }
			if ( (t[i*N+j] & 2) == 2 ) {outputFile<<"<use xlink:href = \"#g\" x = \""<<(j+.5*i)<<"\" y = \""<<i<<"\"/>\n"; }
			if ( (t[i*N+j] & 8) == 8 ) {outputFile<<"<use xlink:href = \"#b\" x = \""<<(j+.5*i)<<"\" y = \""<<i<<"\"/>\n"; }
		}
	}
	outputFile<< "</svg>";
	outputFile.close();
}


void LozengeTiler::DomainToSVG(const domain &d, std::string filename) {
	std::ofstream outputFile(filename.c_str());

	int M = sqrt(d.size()/2);

	outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"433.013\" width=\"500\" viewBox=\"0 0 "<<3/2.0*M<<" "<<3/2.0*M<<"\" preserveAspectRatio=\"none\">\n";
	outputFile << "<defs>\n";
	outputFile << "<g id=\"w\">  <polygon points = \"0,1 .5,0 1,1\" fill=\"lightsteelblue\"/> </g>\n"; //upward triangle
	outputFile << "<g id=\"b\">  <polygon points = \"0,0 1,0 .5,1\" fill=\"slategrey\"/> </g>\n"; //downward triangle
	outputFile << "</defs>\n";

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < 2*M; j++) {
			if ( d[i*M*2+j] == 1 ) {
				if ( j % 2 == 0) outputFile<<"<use xlink:href = \"#b\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
				else outputFile<<"<use xlink:href = \"#w\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
			}
		}
	}

	outputFile<< "</svg>";
	outputFile.close();
}



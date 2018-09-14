#include "DominoTiler.h"
#include "../common/common.h"
#include "../TinyMT/file_reader.h"

/*
 * These are the core functions.
 */

void DominoTiler::Walk(tiling &t, long steps, long seed) {
	int N = std::sqrt(t.size());

	if ( N % 2 != 0 ) std::cout<<"Careful! The tiling has odd dimension."<<std::endl;

	// Break up the tiling into the black and white sub-arrays
	std::vector<char> h_vW((N/2)*N,0);
	std::vector<char> h_vB((N/2)*N,0);

	for(int i=0; i < N ; ++i) {
		for(int j=0; j < N; ++j) {
			if ((i+j)%2==0) h_vB[i*(N/2)+(j/2)] = t[i*N+j];
			else  h_vW[i*(N/2)+(j/2)] = t[i*N+j];
		}
	}

	// Load the arrays to the device.
	cl::Buffer d_vW = cl::Buffer(context, h_vW.begin(), h_vW.end(), CL_FALSE);
	cl::Buffer d_vB = cl::Buffer(context, h_vB.begin(), h_vB.end(), CL_FALSE);


	// Initialize the PRNGs.
	std::mt19937 mt_rand(seed);
	InitTinyMT( cl::EnqueueArgs(queue, cl::NDRange(N*N/2)), tinymtparams, mt_rand());

	for(int i=0; i < steps; ++i) {
		int r = mt_rand()%2;
		if (r==1) {
			RotateTiles( cl::EnqueueArgs( queue, cl::NDRange(N-2,N/2-2)), tinymtparams, d_vB, N, 1);
			UpdateTiles( cl::EnqueueArgs(queue, cl::NDRange(N-2,N/2-2)), d_vW, d_vB, N, 1);
		} else {
			RotateTiles( cl::EnqueueArgs( queue, cl::NDRange(N-2,N/2-2)), tinymtparams, d_vW, N, 0);
			UpdateTiles( cl::EnqueueArgs(queue, cl::NDRange(N-2,N/2-2)), d_vB, d_vW, N, 0);
		}
	}

	// Load back the tilings.
	cl::copy(queue, d_vW, h_vW.begin(), h_vW.end());
	cl::copy(queue, d_vB, h_vB.begin(), h_vB.end());

	for(int i=0; i< N; ++i) {
		for(int j=0; j<N; ++j) {
			if ((i+j)%2==0) t[i*N+j] = h_vB[i*(N/2)+(j/2)];
			else t[i*N+j] = h_vW[i*(N/2)+(j/2)];
		}
	}
}


void DominoTiler::Walk(tiling &t, std::vector<long> steps, std::vector<long> seeds) {
	int N = std::sqrt(t.size());

	if ( N % 2 != 0 ) std::cout<<"Careful! The tiling has odd dimension."<<std::endl;

	std::vector<char> h_vW((N/2)*N,0);
	std::vector<char> h_vB((N/2)*N,0);

	// break up the tiling into the black and white sub-arrays
	for(int i=0; i < N ; ++i) {
		for(int j=0; j < N; ++j) {
			if ((i+j)%2==0)
				h_vB[i*(N/2)+(j/2)] = t[i*N+j];
			else
				h_vW[i*(N/2)+(j/2)] = t[i*N+j];
		}
	}

	cl::Buffer d_vW = cl::Buffer(context, h_vW.begin(), h_vW.end(), CL_FALSE);
	cl::Buffer d_vB = cl::Buffer(context, h_vB.begin(), h_vB.end(), CL_FALSE);

	for (int k = 0; k < steps.size(); ++k) {

		std::mt19937 mt_rand(seeds[k]);
		InitTinyMT( cl::EnqueueArgs(queue, cl::NDRange(N*N/2)), tinymtparams, seeds[k]);

		for(int i=0; i < steps[k]; ++i) {
			int r = mt_rand()%2;
			if (r==1) {
				RotateTiles( cl::EnqueueArgs( queue, cl::NDRange(N-2,N/2-2)), tinymtparams, d_vB, N, 1);
				UpdateTiles( cl::EnqueueArgs(queue, cl::NDRange(N-2,N/2-2)), d_vW, d_vB, N, 1);
			} else {
				RotateTiles( cl::EnqueueArgs( queue, cl::NDRange(N-2,N/2-2)), tinymtparams, d_vW, N, 0);
				UpdateTiles( cl::EnqueueArgs(queue, cl::NDRange(N-2,N/2-2)), d_vB, d_vW, N, 0);
			}
		}
	}

	cl::copy(queue, d_vW, h_vW.begin(), h_vW.end());
	cl::copy(queue, d_vB, h_vB.begin(), h_vB.end());

	for(int i=0; i< N; ++i) {
		for(int j=0; j<N; ++j) {
			if ((i+j)%2==0)
				t[i*N+j] = h_vB[i*(N/2)+(j/2)];
			else
				t[i*N+j] = h_vW[i*(N/2)+(j/2)];
		}
	}
}

void DominoTiler::LoadTinyMT(std::string params, int size) {
	tinymtparams = get_params_buffer(params, context, queue, size);
}


/*
 * Some common domains
 */

domain DominoTiler::Rectangle(int M, int N) {
	int dim = std::max(M,N)+4;
	if (dim%2 == 0) dim++;
	domain tiling(dim*dim,0);
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			int x = j+2; int y = i+2;
			tiling[y*dim+x] = 1;
		}
	}
	return tiling;
}

domain DominoTiler::AztecDiamond(int N) {
	std::vector<int> empty(0,0);
	return AztecRectangle(N, empty);
}

domain DominoTiler::AztecRectangle(int N, const std::vector<int> &remove) {
	int k = remove.size();

	int M =2*N-k;
	std::vector<int> domain(M*M,0); // 2*N-k by 2*N-k domain with 0s everywhere

	for (int i = 0; i < 2*N-k; ++i) {
		for (int j = 0; j < 2*N-k; ++j) {

			if (i == 0 || i == 2*N-k-1) { // top and bottom row
				if (j == (N-k)+(i==2*N-k-1)*k - 1 || j == (N-k)+(i==2*N-k-1)*k) {
					domain[i*M+j] = 1;
				}
			} else {
				if (i<N-k) { // top section
					if (j >= N-k-i-1 && j <= N-k+i) {
						domain[i*M+j] = 1;
					}
				} else if (i > N-1) { // bottom section
					if (j >= i-N+k && j <= 3*N-k-i-1) {
						domain[i*M+j] = 1;
					}
				} else { // middle section
					if (j >= i-(N-k) && j <= i+(N-k)) {
						domain[i*M+j] = 1;
					}
				}
			}
		}
	}
	//std::cout<<"Attempting to remove "<<k<<" points"<<std::endl;
	// note that if the wrong number of points to remove is given the program will inform you of this fact
	// but still create the domain. Let's you create untileable domains if desired.
	for (int i = 0; i<k;++i) { // remove boundary points
		if (remove.empty()) {
			//std::cout<<"No points removed, expecting "<<k<<std::endl;
		}
		else if (remove.size() != k) {
			//std::cout<<remove.size()<<" points removed, expecting "<<k<<std::endl;
		} else {
			int i0 = N-k+remove[i];
			int j0 = remove[i];
			domain[i0*M+j0] = 0;
		}
	}
	//std::cout<<"Done."<<std::endl;

	//padding
	int M2=M+4+(1-M%2); // make sure domain has padding and is of odd size, so that tiling is padded and of even size
	std::vector<int> domain2(M2*M2,0);
	for(int i=0; i<M; ++i) {
		for(int j=0; j<M; ++j) {
			domain2[(i+2)*M2+j+2] = domain[i*M+j];
		}
	}

	return domain2;
}



/*
 * Height functions, and Thurston's algorithm.
 */


// The following variables enumerate the neighboring vertices of a given vertex,
// and the neighboring faces of a given face.
const int neighbors[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
const int neighborfaces[4][2][2] = {{{0,-1},{0,0}},{{-1,-1},{-1,0}},{{-1,0},{0,0}},{{-1,-1},{0,-1}}};
const int pws[4] = {2,1,8,4};
//      1|
// 8 ____|____ 4
//       |
//      2|



tiling DominoTiler::MaxTiling(const domain &d) {
	std::vector<int> vertices = DomainToVertices(d);

	int N = sqrt(vertices.size());

	std::vector<int> hf(N*N, infty); // creates a vector of ints called hf of size N*N with all elements having value INT_MAX

	/* We color even squares white, odd squares black. (Even as in i+j is even.)
	 * Black squares have black vertices NW and SE.
	 * White squares have black vertices NE and SW.
	 *
	 * The height function rules are:
	 * when crossing a black-on-left edge, +1
	 * when crossing a white-on-left edge, -1
	 *
	 * The rules above become:
	 * 		when leaving a black square towards north or south, +1, otherwise -1
	 * 		when leaving a white square towards north or south, -1, otherwise +1
	 */

	// We assume the domain is connected and simply connected.


	// First we find a starting vertex on the boundary of the domain
	std::queue<std::pair<int, int> > bfs;

	bool found = false;

	for (int i = 0; i < N && !found; ++i) {
		for (int j = 0; j < N && !found; ++j) {   // Search through each point in the domain.
			if ( vertices[i*N+j] == 2 ) {   // Check if it is a boundary point.
				found = true;   // If we found a boundary point stop searching,
				bfs.push(std::pair<int, int>(i,j));   // add it to the queue,
				hf[i*N+j] = 0;   // and set its height to 0.
			}
		}
	}

	// Then traverse the boundary, computing the height function along the boundary.
	// At the same time, add boundary points to the BFS queue.
	std::priority_queue< std::pair<std::pair<int, int>,int>, std::vector<std::pair<std::pair<int, int>, int> >, decheightPriority > bfs_pq( ( decheightPriority() ));   // Defines bfs_pq. This contains the boundary points ordered from least height to greatest. So we'll pop the lowest height first.

	// boundary stuff here
	while ( !bfs.empty() )
	{
		int i0 = bfs.front().first; int j0 = bfs.front().second;
		bfs.pop();

		// if (i0+j0)%2 == 0 then +1 for i change
		// if (i0+j0)%2 == 1 then +1 for j change
		for (int nb = 0; nb < 4; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];

			if (vertices[i*N+j] == 2) {
				if((d[(i0+neighborfaces[nb][0][0])*(N-1)+j0+neighborfaces[nb][0][1]] ^ d[(i0+neighborfaces[nb][1][0])*(N-1)+j0+neighborfaces[nb][1][1]])){// check to make sure we are traversing boundary edges
					if ( hf[i*N+j] == infty ) {
						std::pair<int,int> np = std::pair<int, int>(i,j);
						hf[i*N+j] = hf[i0*N+j0] + (nb > 1 ? 1 : -1) * ((i0+j0)%2 == 0 ? 1 : -1);
						bfs.push(np);
						bfs_pq.push(std::pair<std::pair<int,int>, int>(np, hf[i*N+j]));
					} else {
						if (hf[i*N+j] != hf[i0*N+j0] + (nb > 1 ? 1 : -1) * ((i0+j0)%2 == 0 ? 1 : -1)) {
							std::cout<<"!!This region is NOT tileable!! Max, Boundary"<<std::endl;

							return hf;
						}
					}
				}
			}
		}
	}

	// now fill in the height function in the bulk. Start at lowest know vertex and only move upward
	while (!bfs_pq.empty()) {
		int i0 = bfs_pq.top().first.first; int j0 = bfs_pq.top().first.second; bfs_pq.pop();

		for (int nb = 0; nb < 4; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];

			if ( i >= 0 && j >= 0 && i < N && j < N  && vertices[i*N+j] > 0  && ((i0+j0)%2 == 0 ? 1 : -1)*(nb > 1 ? 1 : -1) > 0) {   // Only move in directions that increase height.
				if ( vertices[i*N+j] == 2 && hf[i*N+j] > hf[i0*N+j0] + 1  && vertices[i0*N+j0] != 2) {   // Check height compatibility. Will the catch all cases?
					std::cout<<"!!This region is NOT tileable!! Max, Bulk"<<std::endl;
					return hf;
				} else if ( hf[i*N+j] == infty ) {
					hf[i*N+j] = hf[i0*N+j0] + 1;
					bfs_pq.push(std::pair<std::pair<int,int>, int >(std::pair<int, int>(i,j),hf[i*N+j]));
				}
			}
		}
	}


	return HeightfuncToTiling(hf,d);
}

tiling DominoTiler::MinTiling(const domain &d) {
	std::vector<int> vertices = DomainToVertices(d);

	int N = sqrt(vertices.size());
	std::vector<int> hf(N*N,infty); // creates a vector of ints called hf of size N*N with all elements having value infty


	/* We color even squares black, odd squares white. (Even as in i+j is even.)
	 * Black squares have black vertices NW and SE.
	 * White squares have black vertices NE and SW.
	 *
	 * The height function rules are:
	 * when crossing a black-on-left edge, +1
	 * when crossing a white-on-left edge, -1
	 *
	 * The rules above become:
	 * 		when leaving a black square towards north or south, +1, otherwise -1
	 * 		when leaving a white square towards north or south, -1, otherwise +1
	 */

	// We assume the domain is connected and simply connected.

	// First we find a starting vertex on the boundary of the domain
	std::queue<std::pair<int, int> > bfs;

	bool found = false;

	for (int i = 0; i < N && !found; ++i) {
		for (int j = 0; j < N && !found; ++j) {   // Search through each point in the domain.
			if ( vertices[i*N+j] == 2 ) {   // Check if it is a boundary point.
				found = true;   // If we found a boundary point stop searching,
				bfs.push(std::pair<int, int>(i,j));   // add it to the queue,
				hf[i*N+j] = 0;   // and set its height to 0.
			}
		}
	}

	std::priority_queue< std::pair<std::pair<int, int>,int>, std::vector<std::pair<std::pair<int, int>, int> >, incheightPriority > bfs_pq( ( incheightPriority() ));   // Defines bfs_pq. This contains the boundary points ordered from least height to greatest. So we'll pop the lowest height first.

	// boundary stuff here
	while ( !bfs.empty() )
	{
		int i0 = bfs.front().first; int j0 = bfs.front().second;
		bfs.pop();

		// if (i0+j0)%2 == 0 then +1 for i change
		// if (i0+j0)%2 == 1 then +1 for j change

		for (int nb = 0; nb < 4; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];

			if ( vertices[i*N+j] == 2) {
				if(d[(i0+neighborfaces[nb][0][0])*(N-1)+j0+neighborfaces[nb][0][1]] ^ d[(i0+neighborfaces[nb][1][0])*(N-1)+j0+neighborfaces[nb][1][1]]){ // check that we are traversing a boundary edge
					if ( hf[i*N+j] == infty ) {
						std::pair<int,int> np = std::pair<int, int>(i,j);
						hf[i*N+j] = hf[i0*N+j0] + (nb > 1 ? 1 : -1) * ((i0+j0)%2 == 0 ? 1 : -1);
						bfs.push(np);
						bfs_pq.push(std::pair<std::pair<int,int>, int>(np, hf[i*N+j]));
					} else {
						if (hf[i*N+j] != hf[i0*N+j0] + (nb > 1 ? 1 : -1) * ((i0+j0)%2 == 0 ? 1 : -1)) {
							std::cout<<"!!This region is NOT tileable!! Min, Boundary"<<std::endl;
							return hf;
						}
					}
				}
			}
		}
	}


	// now fill in the height function in the bulk. Start at heighest vertex and only move down.
	while (!bfs_pq.empty()) {
		int i0 = bfs_pq.top().first.first; int j0 = bfs_pq.top().first.second; bfs_pq.pop();

		for (int nb = 0; nb < 4; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];

			if ( i >= 0 && j >= 0 && i < N && j < N  && vertices[i*N+j] > 0  && ((i0+j0)%2 == 0 ? 1 : -1)*(nb > 1 ? 1 : -1) < 0) {   // Only move in directions that decrease height.

				if ( vertices[i*N+j] == 2 && hf[i*N+j] < hf[i0*N+j0] - 1 && hf[i0*N+j0]!=2) {   // Check height compatibility. Will the catch all cases?
					std::cout<<"!!This region is NOT tileable!! Min, Bulk"<<std::endl;  // what about checking with already visted site not on boundary?
					return hf;
				} else if ( hf[i*N+j] == infty ) {
					hf[i*N+j] = hf[i0*N+j0] - 1;
					bfs_pq.push(std::pair<std::pair<int,int>, int >(std::pair<int, int>(i,j),hf[i*N+j]));
				}
			}
		}
	}

	return HeightfuncToTiling(hf,d);
}

tiling DominoTiler::HeightfuncToTiling(const heightfunc &hf, const domain &d) {
	int N = sqrt(hf.size());

	std::vector<int> tiling(N*N,0);

	for(int i0=0; i0<N; ++i0) {
		for(int j0=0; j0<N; ++j0) {
			if ( hf[i0*N+j0] != infty) {

				for (int nb = 0; nb < 4; nb++) {
					int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];
					if ( hf[i*N+j] != infty && (d[(i0 - (nb==1) - (nb>1))*(N-1)+j0 - (nb<=1)-(nb==3)] || d[(i0 - (nb==1))*(N-1)+j0-(nb==3)])) {
						tiling[i0*N+j0] += pws[nb]*std::abs((hf[i*N+j]-hf[i0*N+j0])/2);
					}
				}
			}
		}
	}
	return tiling;
}

heightfunc DominoTiler::TilingToHeightfunc(const tiling &t, const domain &d) {
	int N = sqrt(t.size());

	std::vector<int> vertices = DomainToVertices(d);
	std::vector<int> hf(N*N,infty);
	std::queue<std::pair<int, int> > bfs;

	bool found = false;

	for (int i = 0; i < N && !found; ++i) {
		for (int j = 0; j < N && !found; ++j) {   // Search through each point in the domain.
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

		for (int nb = 0; nb < 4; nb++) {
			int i = i0+neighbors[nb][0]; int j = j0+neighbors[nb][1];
			if ( (t[i0*N+j0] & pws[nb]) == 0) {
				if(vertices[i*N+j] > 0 && hf[i*N+j] == infty) {
					if(d[(i0+neighborfaces[nb][0][0])*(N-1)+j0+neighborfaces[nb][0][1]] || d[(i0+neighborfaces[nb][1][0])*(N-1)+j0+neighborfaces[nb][1][1]]) {
						hf[i*N+j] = hf[i0*N+j0] + (nb > 1 ? 1 : -1) * ((i0+j0)%2 == 0 ? 1 : -1);
						std::pair<int,int> np = std::pair<int, int>(i,j);
						bfs.push(np);
					}
				}
			}
		}
	}

	return hf;
}

domain DominoTiler::TilingToDomain(const tiling &t) {
	int N = sqrt(t.size());
	int M = N-1;
	domain d(M*M, 0);

	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < M; ++j) {
			if ( ((t[i*N+j] & (2 + 8)) != 0) || ( (t[(i+1)*N+(j+1)]  & ( 1 + 4)) != 0) )

				d[i*M+j] = 1;
		}
	}

	return d;
}

std::vector<int> DominoTiler::DomainToVertices(const domain &d) {
	int M = sqrt(d.size());
	int N = M+1;
	std::vector<int> vertices(N*N,0);
	for (int i=0; i<N; ++i) {
		for (int j=0; j<N; ++j) {
			if (i == 0 || j == 0 || i == M || j == M) { // deal with the edge of the domain
				if (d[(i - (i==M))*M + j - (j==M)] || d[(i-1+(i==0))*M + j-(j==M)] || d[(i-(i==M))*M + j-1+(j==0)] || d[((i-1+(i==0))*M + j-1+(j==0))]) {
					vertices[i*N+j] = 2;
				}
			} else {
				if (d[(i-1)*M + j-1] && d[(i-1)*M + j] && d[(i)*M + j-1] && d[(i)*M + j]) { // check if interior vertex
					vertices[i*N+j] = 1;
				} else if (d[(i-1)*M + j-1] || d[(i-1)*M + j] || d[(i)*M + j-1] || d[(i)*M + j]) { // if not check if boundary vertex
					vertices[i*N+j] = 2;
				}
			}
		}
	}
	return vertices;
}

/*
 * some conventions for drawing:
 * vertex (i,j) of the domain is embedded at (i,j)
 * face (i,j) has center (i+.5, j+.5)
 *
 * */

void DominoTiler::DomainToSVG(const domain &d, std::string filename) {
	std::ofstream outputFile(filename.c_str());

	int M = sqrt(d.size());

	outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"500\" width=\"500\" viewBox=\"0 0 "<<M<<" "<<M<<"\">\n";
	outputFile << "<defs>\n";
	outputFile << "<g id=\"b\">  <polygon points = \"-.5,-.5 .5,-.5 .5,.5 -.5,.5\" fill=\"blue\"/> </g>\n";
	outputFile << "<g id=\"w\">  <polygon points = \"-.5,-.5 .5,-.5 .5,.5 -.5,.5\" fill=\"grey\"/> </g>\n";
	outputFile << "</defs>\n";

	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < M; ++j) {

			if ( d[i*M+j] == 1) {
				double x = (j+.5);
				double y = (i+.5);

				if ( (i+j)% 2 == 0)
					outputFile<<"<use xlink:href = \"#b\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
				else
					outputFile<<"<use xlink:href = \"#w\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
			}
		}
	}

	outputFile<< "</svg>";
	outputFile.close();
}

void DominoTiler::DimerToSVG(const tiling &t, std::string filename) {
	domain d = TilingToDomain(t);
	DimerToSVG(t, d, filename);
}

void DominoTiler::DimerToSVG(const tiling &t, const domain &d, std::string filename) {

	std::ofstream outputFile(filename.c_str());

	int N = sqrt(t.size());
	int M = N-1;

	outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"500\" width=\"500\" viewBox=\"0 0 "<<M<<" "<<M<<"\">\n";
	outputFile << "<defs>\n";
	outputFile << "<g id=\"hd\">  <polyline points= \"-.5,0 .5,0\" stroke=\"red\" stroke-width=\".05\"/> </g>\n";
	outputFile << "<g id=\"he\">  <polyline points= \"-.5,0 .5,0\" stroke=\"black\" stroke-width=\".01\"/> </g>\n";
	outputFile << "<g id=\"vd\">  <polyline points= \"0,-.5 0,.5\" stroke=\"red\" stroke-width=\".05\"/> </g>\n";
	outputFile << "<g id=\"ve\">  <polyline points= \"0,-.5 0,.5\" stroke=\"black\" stroke-width=\".01\"/> </g>\n";

	outputFile << "<g id=\"b\">  <circle cx=\"0\" cy=\"0\" r=\".1\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/> </g> \n";
	outputFile << "<g id=\"w\">  <circle cx=\"0\" cy=\"0\" r=\".1\" stroke=\"black\" stroke-width=\".02\" fill=\"white\" /> </g>\n";
	outputFile << "</defs>\n";
	//draw dimers

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double x = j; double y = i;
			if ( (t[i*N+j]  & 1) == 1) { outputFile<<"<use xlink:href=\"#hd\" x=\""<<x<<"\" y=\""<<y-.5<<"\"/>\n"; }
			if ( (t[i*N+j]  & 4) == 4) { outputFile<<"<use xlink:href=\"#vd\" x=\""<<x-.5<<"\" y=\""<<y<<"\"/>\n"; }
		}
	}


	for (int i = 1; i < M; ++i) {
		for (int j = 1; j < M; ++j) {
			if (d[i*M+j] == 1 ) {
				double x = (j+.5);
				double y = (i+.5);

				if ((i+j)%2 == 1) {
					if ( d[i*M +j+1] == 1) {
						outputFile<<"<use xlink:href=\"#he\" x=\""<<x+.5<<"\" y=\""<<y<<"\"/>\n";

					}
					if ( d[i*M +j-1] == 1) {
						outputFile<<"<use xlink:href=\"#he\" x=\""<<x-.5<<"\" y=\""<<y<<"\"/>\n";
					}
				}
				if ( d[(i-1)*M +j] == 1) {
					outputFile<<"<use xlink:href=\"#ve\" x=\""<<x<<"\" y=\""<<y-.5<<"\"/>\n";
				}

				if ( d[(i+1)*M +j] == 1) {
					outputFile<<"<use xlink:href=\"#ve\" x=\""<<x<<"\" y=\""<<y+.5<<"\"/>\n";
				}
			}
		}
	}

	for (int i = 1; i < M; ++i) {
		for (int j = 1; j < M; ++j) {
			if (d[i*M+j] == 1 ) {
				double x = (j+.5);
				double y = (i+.5);
				if ( (i+j)%2 == 0)
					outputFile<<"<use xlink:href = \"#b\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
				else
					outputFile<<"<use xlink:href = \"#w\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";

			}}}
	outputFile<< "</svg>";
	outputFile.close();
}


void DominoTiler::TilingToSVG(const tiling &t, std::string filename) {
	std::ofstream outputFile(filename.c_str());

	int N = sqrt(t.size());
	outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"500\" width=\"500\" viewBox=\"0 0 "<<(N-1)<<" "<<(N-1)<<"\">\n";
	outputFile << "<defs>\n";
	outputFile << "<g id=\"h1\">  <polygon points = \"-1,0 -1,-1 1,-1 1,0\" fill=\"white\" stroke=\"black\" stroke-width=\".05\"/> </g>\n";
	outputFile << "<g id=\"v1\">  <polygon points = \"-1,-1 0,-1 0,1 -1,1\" fill=\"white\" stroke=\"black\" stroke-width=\".05\"/> </g>\n";
	outputFile << "<g id=\"h2\">  <polygon points = \"-1,0 -1,-1 1,-1 1,0\" fill=\"white\" stroke=\"black\" stroke-width=\".05\"/> </g>\n";
	outputFile << "<g id=\"v2\">  <polygon points = \"-1,-1 0,-1 0,1 -1,1\" fill=\"white\" stroke=\"black\" stroke-width=\".05\"/> </g>\n";

	outputFile << "</defs>\n";

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double x = j;
			double y = i;
			if ( (i +j )%2 == 1) {
				if ( (t[i*N+j]  & 1) == 1) outputFile<<"<use xlink:href = \"#h1\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
				if ( (t[i*N+j]  & 4) == 4) outputFile<<"<use xlink:href = \"#v1\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
			} else {
				if ( (t[i*N+j]  & 1) == 1) outputFile<<"<use xlink:href = \"#h2\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
				if ( (t[i*N+j]  & 4) == 4) outputFile<<"<use xlink:href = \"#v2\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
			}}
	}
	outputFile<< "</svg>";
	outputFile.close();
}

void DominoTiler::MayaToSVG(const tiling &t, std::string filename) {
	std::ofstream outputFile(filename.c_str());

	int N = sqrt(t.size());
	outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"500\" width=\"500\" viewBox=\"0 0 "<<(N-1)<<" "<<(N-1)<<"\">\n";
	outputFile << "<defs>\n";
	outputFile << "<g id=\"h1\">";
	outputFile << "<circle cx=\"-.5\" cy=\"-.5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/> <circle cx=\".5\" cy=\"-.5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/>";
	//outputFile << "<polygon points = \"-1,0 -1,-1 1,-1 1,0\" fill=\"none\" stroke=\"black\" stroke-width=\".05\"/>";
	outputFile << "</g>\n";
	outputFile << "<g id=\"h2\">";
	outputFile << "<circle cx=\"-.5\" cy=\"-.5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"white\"/> <circle cx=\".5\" cy=\"-.5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"white\"/>";
	//outputFile << "<polygon points = \"-1,0 -1,-1 1,-1 1,0\" fill=\"none\" stroke=\"black\" stroke-width=\".05\"/>";
	outputFile << "</g>\n";
	outputFile << "<g id=\"v1\">";
	outputFile << "<circle cx=\"-.5\" cy=\"-.5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/> <circle cx=\"-.5\" cy=\".5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/>";
	//outputFile << "<polygon points = \"-1,-1 0,-1 0,1 -1,1\" fill=\"none\" stroke=\"black\" stroke-width=\".05\"/>";
	outputFile << "</g>\n";
	outputFile << "<g id=\"v2\">";
	outputFile << "<circle cx=\"-.5\" cy=\"-.5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"white\"/> <circle cx=\"-.5\" cy=\".5\" r=\".4\" stroke=\"black\" stroke-width=\".02\" fill=\"white\"/>";
	//outputFile << "<polygon points = \"-1,-1 0,-1 0,1 -1,1\" fill=\"none\" stroke=\"black\" stroke-width=\".05\"/>";
	outputFile << "</g>\n";
	outputFile << "</defs>\n";

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double x = j;
			double y = i;
			if ( (i +j )%2 == 1) {
				if ( (t[i*N+j]  & 1) == 1) outputFile<<"<use xlink:href = \"#h1\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
				if ( (t[i*N+j]  & 4) == 4) outputFile<<"<use xlink:href = \"#v1\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
			} else {
				if ( (t[i*N+j]  & 1) == 1) outputFile<<"<use xlink:href = \"#h2\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
				if ( (t[i*N+j]  & 4) == 4) outputFile<<"<use xlink:href = \"#v2\" x = \""<<x<<"\" y = \""<<y<<"\"/>\n";
			}
		}
	}
	outputFile<< "</svg>";
	outputFile.close();
}

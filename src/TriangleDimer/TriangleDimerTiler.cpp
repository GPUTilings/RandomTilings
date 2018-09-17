//
//  TriangleDimerTiler.cpp
//
//
//  Created by Ananth, David
//


#include "TriangleDimerTiler.h"
#include "../common/common.h"
#include "../TinyMT/file_reader.h"




void TriangleDimerTiler::LoadTinyMT(std::string params, int size) {
	tinymtparams = get_params_buffer(params, context, queue, size);
}

std::vector<int> TriangleDimerTiler::DimerIndicator(tiling &t, int c) {
    int N = std::sqrt(t.size());
    int M = N/2;
    std::vector<int> D(M*M,0);
    std::vector<int> DD(M*M,0);
    
    if (c==1) { // Left
        for(int i=0; i < N ; ++i) {
            for(int j=0; j < N; ++j) {
                if(i%2 == 0) {
                    if(j%2 == 1) {
                        D[(i/2)*M+j/2] = t[i*N+j]; // elements of orientation L
                    }
                }
            }
        }
        for(int i=0; i<M; ++i) {
            for(int j=0; j<M; ++j) {
                if((D[i*M+j]&8) == 8) {
                    DD[i*M+j] = 1;
                }
            }
        }
        
    } else if (c==2) { // Right
        for(int i=0; i < N ; ++i) {
            for(int j=0; j < N; ++j) {
                if(i%2 == 0) {
                    if(j%2 == 1) {
                        D[(i/2)*M+j/2] = t[i*N+j]; // elements of orientation R
                    }
                }
            }
        }
        for(int i=0; i<M; ++i) {
            for(int j=0; j<M; ++j) {
                if((D[i*M+j]&4) == 4) {
                    DD[i*M+j] = 1;
                }
            }
        }
    } else if (c==0) { // Horizontal
        for(int i=0; i < N ; ++i) {
            for(int j=0; j < N; ++j) {
                if(i%2 == 1) {
                    if(j%2 == 1) {
                        D[(i/2)*M+j/2] = t[i*N+j]; // elements of orientation H
                    }
                }
            }
        }
        for(int i=0; i<M; ++i) {
            for(int j=0; j<M; ++j) {
                if((D[i*M+j]&1) == 1) {
                    DD[i*M+j] = 1;
                }
            }
        }
    }
    
    return DD;
    
    
    
}

void TriangleDimerTiler::Walk(tiling &t, int steps, long seed) {
    int N = std::sqrt(t.size());
    int M = N/2;
    
    tiling tTri = TriangleDimerTiler::TilingToTriangleFlips(t);
    tiling tB = TriangleDimerTiler::TilingToButterflyFlips(t);
    
    // initialize vectors
    // Lozenge states
    std::vector<int> h_eR(M*M,0);
    std::vector<int> h_eL(M*M,0);
    std::vector<int> h_eH(M*M,0);
    // Triangle states
    std::vector<int> h_tU(M*M,0);
    std::vector<int> h_tD(M*M,0);
    // Butterfly states
    std::vector<int> h_bR(M*M,0);
    std::vector<int> h_bL(M*M,0);
    std::vector<int> h_bH(M*M,0);
    
    // fill host vectors
    for(int i=0; i < N ; ++i) {
        for(int j=0; j < N; ++j) {
            if(i%2 == 0) {
                if(j%2 == 1) {
                    h_eH[(i/2)*M+j/2] = t[i*N+j]; // Lozenge-type orientation H
                    h_bH[(i/2)*M+j/2] = tB[i*N+j]; // Butterfly-type orientation H
                }
            } else {
                if(j%2 == 0) {
                    h_eL[(i/2)*M+j/2] = t[i*N+j]; //  Lozenge-type orientation L
                    h_bL[(i/2)*M+j/2] = tB[i*N+j]; // Butterfly-type orientation L
                } else {
                    h_eR[(i/2)*M+j/2] = t[i*N+j]; // Lozenge-type orientation R
                    h_bR[(i/2)*M+j/2] = tB[i*N+j]; // Butterfly-type orientation R
                }
            }
        }
    }
    for(int i=0; i < N/2; ++i) {
        for(int j=0; j < N; ++j) {
            if(j%2 == 0) {
                h_tD[i*M+j/2] = tTri[i*N+j]; // Triangle-type orientation D
            } else {
                h_tU[i*M+j/2] = tTri[i*N+j]; // Triangle-type orientation U
            }
        }
    }
    
    
    
//    std::cout<<"Lozenge Type:"<<std::endl;
//    for(int i=0; i<N; ++i) {
//        for(int j=0; j<N; ++j) {
//            std::cout<<t[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"H:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_eH[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"L:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_eL[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"R:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_eR[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//
//    std::cout<<"Triangle type:"<<std::endl;
//    for(int i=0; i < N/2; ++i) {
//        for(int j=0; j < N; ++j) {
//            std::cout<<tTri[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
//    std::cout<<"Triangle type U:"<<std::endl;
//    for(int i=0; i < M; ++i) {
//        for(int j=0; j < M; ++j) {
//            std::cout<<h_tU[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
//    std::cout<<"Triangle type D:"<<std::endl;
//    for(int i=0; i < M; ++i) {
//        for(int j=0; j < M; ++j) {
//            std::cout<<h_tD[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
    
//    std::cout<<"Butterfly Type:"<<std::endl;
//    for(int i=0; i<N; ++i) {
//        for(int j=0; j<N; ++j) {
//            std::cout<<tB[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"H:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_bH[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"L:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_bL[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"R:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_bR[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
    
    // device arrays
    cl::Buffer d_eR = cl::Buffer(context, h_eR.begin(), h_eR.end(), CL_FALSE);
    cl::Buffer d_eL = cl::Buffer(context, h_eL.begin(), h_eL.end(), CL_FALSE);
    cl::Buffer d_eH = cl::Buffer(context, h_eH.begin(), h_eH.end(), CL_FALSE);
    cl::Buffer d_tD = cl::Buffer(context, h_tD.begin(), h_tD.end(), CL_FALSE);
    cl::Buffer d_tU = cl::Buffer(context, h_tU.begin(), h_tU.end(), CL_FALSE);
    cl::Buffer d_bR = cl::Buffer(context, h_bR.begin(), h_bR.end(), CL_FALSE);
    cl::Buffer d_bL = cl::Buffer(context, h_bL.begin(), h_bL.end(), CL_FALSE);
    cl::Buffer d_bH = cl::Buffer(context, h_bH.begin(), h_bH.end(), CL_FALSE);
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> d1(0,2);
    std::uniform_int_distribution<int> d2(0,1);
    generator.seed(seed);
    
    InitTinyMT( cl::EnqueueArgs(queue, cl::NDRange(M*M)), tinymtparams, seed );
    
    for(int i=0; i < steps; ++i) {
        //run kernels
        int rFlipType = d1(generator); // each type of flip has equal probability
        int r1 = d1(generator);
        int r2 = d2(generator);
        
        if(rFlipType == 0) { // Lozenge flips
            if(r1==0) { //Horizontal Lozenges
                // Rotate horizontal
                RotateLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_eH, M, r1, r2);
                // Update those horizontal that rotated
                UpdateLozengesFlipped(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eH, M, r1, r2);
                // Update the rest of the lozenges
                UpdateLozenges0(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_eH ,d_eL ,d_eR , M);
                // Update Up triangles
                UpdateTriangleUFromLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eH, d_eL, d_tU, M);
                // Update down triangles from up triangles
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tU , d_tD, M, 0);
                // Update butterfly from lozenges
                UpdateButterflysHFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                UpdateButterflysLFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysRFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
            } else if(r1==1) { //Left Lozenges
                RotateLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_eL, M, r1, r2);
                UpdateLozengesFlipped(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eL, M, r1, r2);
                UpdateLozenges1(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_eL ,d_eR ,d_eH , M);
                UpdateTriangleUFromLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eH, d_eL, d_tU, M);
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tU , d_tD, M, 0);
                UpdateButterflysHFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                UpdateButterflysLFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysRFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
            } else {//Right Lozenges
                RotateLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_eR, M, r1, r2);
                UpdateLozengesFlipped(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eR, M, r1, r2);
                UpdateLozenges2(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_eR ,d_eH ,d_eL , M);
                UpdateTriangleUFromLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eH, d_eL, d_tU, M);
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tU , d_tD, M, 0);
                UpdateButterflysHFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                UpdateButterflysLFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysRFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
            }
        } else if(rFlipType == 1) { // Triangle Flips
            if(r2 == 1) { // Down triangles
                // Rotate triangles
                RotateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_tD, M, r1);
                // Update the triangles that flipped
                UpdateTrianglesFlipped1(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tD, M, r1);
                // Update the rest of the triangles
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tD , d_tU, M, r2);
                // Update Lozenge from triangles
                UpdateLozengeHFromTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tU, d_tD, d_eH, M);
                UpdateLozengeLFromTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tU, d_tD, d_eL, M);
                UpdateLozengeRFromTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tU, d_tD, d_eR, M);
                // Update Butterfly from lozenge
                UpdateButterflysHFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                UpdateButterflysLFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysRFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
            } else { // Up triangles
                RotateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_tU, M, r1);
                UpdateTrianglesFlipped0(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tU, M, r1);
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tU , d_tD, M, r2);
                UpdateLozengeHFromTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tU, d_tD, d_eH, M);
                UpdateLozengeLFromTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tU, d_tD, d_eL, M);
                UpdateLozengeRFromTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_tU, d_tD, d_eR, M);
                UpdateButterflysHFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                UpdateButterflysLFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysRFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
            }
        } else { // Butterfly Flips
            int p1 = d1(generator);
            int p2 = d2(generator);
            
            if(r1 == 0) { // Horizontal Butterflys
                // Rotate Butterflys
                RotateButterflys(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_bH, M, r1, p1, p2);
                // Series of kernels to update all the horizontal butterflys
                UpdateButterflysFlippedH1(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bH,M,p1,p2);
                UpdateButterflysFlippedH21(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bH,M,p1);
                UpdateButterflysFlippedH22(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bH,M,p1);
                UpdateButterflysFlippedH23(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bH,M,p1);
                // Update lozenges from horizontal butterflys
                UpdateLozengeFromButterflysH(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                // Update the rest of the butterflys from lozenge
                UpdateButterflysLFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysRFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
                // Update triangles from lozenge
                UpdateTriangleUFromLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eH, d_eL, d_tU, M);
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tU , d_tD, M, 0);
            } else if (r1 == 1) { // Left Butterflys
                RotateButterflys(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_bL, M, r1, p1, p2);
                UpdateButterflysFlippedL1(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bL,M,p1,p2);
                UpdateButterflysFlippedL21(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bL,M,p1);
                UpdateButterflysFlippedL22(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bL,M,p1);
                UpdateButterflysFlippedL23(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bL,M,p1);
                UpdateLozengeFromButterflysL(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysHFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                UpdateButterflysRFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
                UpdateTriangleUFromLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eH, d_eL, d_tU, M);
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tU , d_tD, M, 0);
            } else { // Right Butterflys
                RotateButterflys(cl::EnqueueArgs( queue, cl::NDRange(M,M)), tinymtparams, d_bR, M, r1, p1, p2);
                UpdateButterflysFlippedR1(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bR,M,p1,p2);
                UpdateButterflysFlippedR21(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bR,M,p1);
                UpdateButterflysFlippedR22(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bR,M,p1);
                UpdateButterflysFlippedR23(cl::EnqueueArgs(queue, cl::NDRange(M,M)),d_bR,M,p1);
                UpdateLozengeFromButterflysR(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bR,d_eH,d_eL,d_eR,M);
                UpdateButterflysLFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bL,d_eH,d_eL,d_eR,M);
                UpdateButterflysHFromLozenge(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_bH,d_eH,d_eL,d_eR,M);
                UpdateTriangleUFromLozenges(cl::EnqueueArgs( queue, cl::NDRange(M,M)), d_eH, d_eL, d_tU, M);
                UpdateTriangles(cl::EnqueueArgs( queue, cl::NDRange(M,M)),d_tU , d_tD, M, 0);
            }
        }
        
    }
    
    cl::copy(queue, d_eR, h_eR.begin(), h_eR.end());
    cl::copy(queue, d_eL, h_eL.begin(), h_eL.end());
    cl::copy(queue, d_eH, h_eH.begin(), h_eH.end());
    cl::copy(queue, d_tU, h_tU.begin(), h_tU.end());
    cl::copy(queue, d_tD, h_tD.begin(), h_tD.end());
    cl::copy(queue, d_bR, h_bR.begin(), h_bR.end());
    cl::copy(queue, d_bL, h_bL.begin(), h_bL.end());
    cl::copy(queue, d_bH, h_bH.begin(), h_bH.end());

    
    for(int i=0; i<N; ++i) { // combine vectors back into tile state
        for(int j=0; j<N; ++j) {
            if(i%2 == 0) {
                if(j%2 == 1) {
                    t[i*N+j] = h_eH[(i/2)*M+j/2];
                }
            } else {
                if(j%2 == 0) {
                    t[i*N+j] = h_eL[(i/2)*M+j/2];
                } else {
                    t[i*N+j] = h_eR[(i/2)*M+j/2];
                }
            }
        }
    }
    
    for(int i=0; i < M; ++i) {
        for(int j=0; j < M; ++j) {
                tTri[i*N+2*j] = h_tD[i*M+j];
                tTri[i*N+2*j+1] = h_tU[i*M+j];
        }
    }
    
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            if(i%2 == 0) {
                if(j%2 == 1) {
                    tB[i*N+j] = h_bH[(i/2)*M+j/2];
                }
            } else {
                if(j%2 == 0) {
                    tB[i*N+j] = h_bL[(i/2)*M+j/2];
                } else {
                    tB[i*N+j] = h_bR[(i/2)*M+j/2];
                }
            }
        }
    }
    
//    std::cout<<"Butterfly Type:"<<std::endl;
//    for(int i=0; i<N; ++i) {
//        for(int j=0; j<N; ++j) {
//            std::cout<<tB[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"H:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_bH[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"L:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_bL[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"R:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_bR[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
    
//    std::cout<<"Triangle type:"<<std::endl;
//    for(int i=0; i < N/2; ++i) {
//        for(int j=0; j < N; ++j) {
//            std::cout<<tTri[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
//    std::cout<<"Triangle type U:"<<std::endl;
//    for(int i=0; i < M; ++i) {
//        for(int j=0; j < M; ++j) {
//            std::cout<<h_tU[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
//    std::cout<<"Triangle type D:"<<std::endl;
//    for(int i=0; i < M; ++i) {
//        for(int j=0; j < M; ++j) {
//            std::cout<<h_tD[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
//    std::cout<<"H:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_eH[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"L:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_eL[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"R:"<<std::endl;
//    for(int i=0; i<M; ++i) {
//        for(int j=0; j<M; ++j) {
//            std::cout<<h_eR[i*M+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"T:"<<std::endl;
//    for(int i=0; i<N; ++i) {
//        for(int j=0; j<N; ++j) {
//            std::cout<<t[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }

}

tiling TriangleDimerTiler::IceCreamCone(int N, int M) {
    // Creates an equilateral triangle with side length N edges, pointed downward, with another equilateral triangle of side length M cut out of the top left corner.
    if(M>N-1) {
        std::cout<<"Cutting out too big a chunk."<<std::endl;
    }
    if(((((N+1)*(N+2))/2 +(M*(M+1))/2)% 2) != 0) {
        std::cout<<"No dimer cover exists."<<std::endl;
        // When M=0 but the domain is not tileable, the below will act as though the vertex in the top right corner was removed and construct a viable dimer cover of that domain.
    }
    
    int eSize = N+4;
    int tSize = 2*eSize;
    int k = (((N+1)*(N+2))/2)%2;
    
    std::vector<int> h(eSize*eSize,0); //horizontal edges
    std::vector<int> l(eSize*eSize,0); //up-left edges
    std::vector<int> r(eSize*eSize,0); //up-right edges
    tiling t(tSize*tSize,0);
    
    // 1 if dimer, 0 if no dimer
    for(int i=2; i<eSize-2; ++i) {
        for(int j=2; j<eSize-i; ++j) {
            if(i<M+2) {
                if((i+j)%2==M%2 && (M!=0 && j>M+3-i)) {
                    h[i*eSize+j] = 1;
                }
                if((N-M)%2==0) {
                    if((i%2)==0 && j==eSize-i-1) {
                        r[i*eSize+j] = 1;
                    }
                }
            } else {
                if(j%2 == 0) {
                    h[i*eSize+j] = 1;
                }
                if(i%4 == (3+N%2+2*k)%4 && j==eSize-i-1) {
                    h[i*eSize+j] = 0;
                    l[i*eSize+j] = 1;
                }
                if(i%4 == (2+N%2+2*k)%4 && j==eSize-i-1) {
                    r[i*eSize+j] = 1;
                }
            }
        }
    }
    // construct tile state from dimers
    for(int i=3; i<tSize-1; ++i) {
        for(int j=3; j<tSize-1; ++j) {
            if(i%2==0) {
                if(j%2==1) {
                    t[i*tSize+j] = r[((i/2)-1)*eSize+j/2] + 2*l[((i/2)-1)*eSize+j/2+1] + 4*r[(i/2)*eSize+j/2] + 8*l[(i/2)*eSize+j/2];
                }
            } else {
                if(j%2==0) {
                    t[i*tSize+j] = h[(i/2)*eSize+j/2] +2*r[(i/2)*eSize+j/2] + 4*h[((i/2)+1)*eSize+j/2-1] + 8*r[(i/2)*eSize+j/2-1];
                } else {
                    t[i*tSize+j] = h[(i/2)*eSize+j/2] +2*l[(i/2)*eSize+j/2+1] + 4*h[((i/2)+1)*eSize+j/2] + 8*l[(i/2)*eSize+j/2];
                }
            }
        }
    }
    
    
//    std::cout<<"h:"<<std::endl;
//    for(int i=0; i<eSize; ++i) {
//        for(int j=0; j<eSize; ++j) {
//            std::cout<<h[i*eSize+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"l:"<<std::endl;
//    for(int i=0; i<eSize; ++i) {
//        for(int j=0; j<eSize; ++j) {
//            std::cout<<l[i*eSize+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"r:"<<std::endl;
//    for(int i=0; i<eSize; ++i) {
//        for(int j=0; j<eSize; ++j) {
//            std::cout<<r[i*eSize+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<"t:"<<std::endl;
//    for(int i=0; i<tSize; ++i) {
//        for(int j=0; j<tSize; ++j) {
//            std::cout<<t[i*tSize+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
//    std::cout<<"tSize = "<<tSize<<std::endl;
    
    return t;
    
}

tiling TriangleDimerTiler::TestButterfly(int type) {
    // Creates a basic butterfly of chosen orientation. Used to test butterfly rotate/update kernels.
    int H = 0; int L = 1; int R = 2;
    int eSize = 4+4;
    int tSize = 2*eSize;
    
    std::vector<int> h(eSize*eSize,0); //horizontal edges
    std::vector<int> l(eSize*eSize,0); //up-left edges
    std::vector<int> r(eSize*eSize,0); //up-right edges
    tiling t(tSize*tSize,0);
    
    if (type == H) {
        h[2*eSize+3] = 1;
        h[4*eSize+3] = 1;
        r[2*eSize+4] = 1;
        r[3*eSize+2] = 1;
    } else if (type == L) {
        r[2*eSize+3] = 1;
        r[4*eSize+2] = 1;
        l[3*eSize+2] = 1;
        l[3*eSize+4] = 1;
    } else if (type == R) {
        l[2*eSize+4] = 1;
        l[4*eSize+3] = 1;
        r[3*eSize+2] = 1;
        r[3*eSize+4] = 1;
    } else {
        std::cout<<"Please choose types 0 (H), 1 (L), or 2 (R)."<<std::endl;
        return t;
    }
    
    // construct tile state from dimers
    for(int i=3; i<tSize-1; ++i) {
        for(int j=3; j<tSize-1; ++j) {
            if(i%2==0) {
                if(j%2==1) {
                    t[i*tSize+j] = r[((i/2)-1)*eSize+j/2] + 2*l[((i/2)-1)*eSize+j/2+1] + 4*r[(i/2)*eSize+j/2] + 8*l[(i/2)*eSize+j/2];
                }
            } else {
                if(j%2==0) {
                    t[i*tSize+j] = h[(i/2)*eSize+j/2] +2*r[(i/2)*eSize+j/2] + 4*h[((i/2)+1)*eSize+j/2-1] + 8*r[(i/2)*eSize+j/2-1];
                } else {
                    t[i*tSize+j] = h[(i/2)*eSize+j/2] +2*l[(i/2)*eSize+j/2+1] + 4*h[((i/2)+1)*eSize+j/2] + 8*l[(i/2)*eSize+j/2];
                }
            }
        }
    }
    
    return t;
}

std::vector<int> TriangleDimerTiler::TilingToTriangleFlips(tiling &t) {
    int N = sqrt(t.size());
    std::vector<int> triFlips((N/2)*N,0);
    
    // assumes padding in tiling
    for(int i = 1; i < (N/2)-1; ++i) {
        for(int j = 1; j < N-1; ++j) {
            if(j%2==0) { // down triangles
                triFlips[i*N+j] = ((t[(2*i+1)*N+j] & 4)/4 + (t[(2*i+1)*N+j] & 8)/4 + ((t[(2*i+1)*N+j+1] & 8)/8)*3) + 4*((t[(2*i+1)*N+j+1] & 4)/4 + (t[(2*i+1)*N+j] & 2) + ((t[(2*i+1)*N+j+1] & 2)/2)*3) + 16*((t[2*i*N+j+1] & 1)*2 + (t[(2*i+1)*N+j] & 1) + ((t[2*i*N+j+1] & 2)/2)*3);
            } else { // up triangle
                triFlips[i*N+j] = ((t[(2*i+1)*N + j] & 1) + (t[(2*i+1)*N + j] & 8)/4 + ((t[(2*i+1)*N+j+1] & 8)/8)*3) + 4*((t[(2*i+1)*N + j+1] & 1) +(t[(2*i+1)*N + j] & 2)  +((t[(2*i+1)*N + j+1] & 2)/2)*3) + 16*((t[(2*i+2)*N + j] & 8)/4 + ((t[(2*i+2)*N + j] & 4)/4)*3 + (t[(2*i+1)*N + j] & 4)/4);
            }
            
        }
    }
    
//    std::cout<<"N = "<<N<<std::endl;
//    std::cout<<"Triangle states:"<<std::endl;
//    for(int i=0; i<N/2; ++i) {
//        for(int j=0; j<N; ++j) {
//            std::cout<<triFlips[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
    
    return triFlips;
}

std::vector<int> TriangleDimerTiler::TilingToButterflyFlips(tiling &t) {
    int N = sqrt(t.size());
    tiling tB(N*N,0);
    
    int f1; int f4; int f16; int f64; int c;
    for(int i=2; i<N-2; ++i) {
        for(int j=2; j<N-2; ++j) {
            if (i%2 == 0) {
                if(j%2 == 1) { // H
                    f1 = (t[(i-1)*N + j] & 8)/8 + (t[(i-1)*N + j] & 1)*2 + (t[i*N+j] & 1)*3;
                    f4 = (t[(i-1)*N + j+1] & 1) + (t[(i-1)*N + j+1] & 2) + ((t[i*N+j] & 2)/2)*3;
                    f16 = (t[(i+1)*N + j] & 2)/2 + (t[(i+1)*N + j] & 4)/2 + ((t[i*N+j] & 4)/4)*3;
                    f64 = (t[(i+1)*N + j-1] & 4)/4 + (t[(i+1)*N + j-1] & 8)/4 + ((t[i*N+j] & 8)/8)*3;
                    c = (t[(i-1)*N + j+1] & 4)/4;
                    tB[i*N+j] = f1 + 4*f4 + 16*f16 + 64*f64 + 256*c;
                }
            } else {
                if(j%2 == 0) { // L
                    f1 = (t[(i-1)*N+j+1] & 1) + (t[(i-1)*N+j+1] & 2) + (t[i*N+j] & 1)*3;
                    f4 = (t[i*N+j+1] & 2)/2 + (t[i*N+j+1] & 4)/2 + ((t[i*N+j] & 2)/2)*3;
                    f16 = (t[(i+1)*N+j-1] & 4)/4 + (t[(i+1)*N+j-1] & 8)/4 + ((t[i*N+j] & 4)/4)*3;
                    f64 = (t[i*N+j-1] & 8)/8 + (t[i*N+j-1] & 1)*2 + ((t[i*N+j] & 8)/8)*3;
                    c = (t[i*N+j+1] & 8)/8;
                    tB[i*N+j] = f1 + 4*f4 + 16*f16 + 64*f64 + 256*c;
                } else { // R
                    f1 = (t[(i-1)*N+j] & 1) + (t[(i-1)*N+j] & 2) + (t[i*N+j] & 1)*3;
                    f4 = (t[i*N+j+1] & 1) + (t[i*N+j+1] & 2) + ((t[i*N+j] & 2)/2)*3;
                    f16 = (t[(i+1)*N+j] & 4)/4 + (t[(i+1)*N+j] & 8)/4 + ((t[i*N+j] & 4)/4)*3;
                    f64 = (t[i*N+j-1] & 4)/4 + (t[i*N+j-1] & 8)/4 + ((t[i*N+j] & 8)/8)*3;
                    c = (t[i*N+j+1] & 8)/8;
                    tB[i*N+j] = f1 + 4*f4 + 16*f16 + 64*f64 + 256*c;
                }
            }
        }
    }
    
//    std::cout<<"Butterfly states:"<<std::endl;
//    for(int i=0; i<N; ++i) {
//        for(int j=0; j<N; ++j) {
//            std::cout<<tB[i*N+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//    std::cout<<std::endl;
    
    return tB;
}

std::vector<int> TriangleDimerTiler::TilingToVertices(tiling &t) {
    // From tile state create array of vertces in the domain.
    int N = sqrt(t.size());
    int M = N/2+1;
    
    std::vector<int> v(M*M,0);
    
    for(int i=2; i<M-2; ++i) {
        for(int j=2; j<M-2; ++j) {
            if((t[2*(i-1)*N+2*j+1] & 4) == 0 && (t[2*(i-1)*N+2*j+1] & 8) == 0  && (t[2*(i+1)*N+2*j-1] & 2) == 0 && (t[2*(i+1)*N+2*j-1] & 1) == 0 && (t[(2*i-1)*N+2*j] & 4) == 0 && (t[(2*i-1)*N+2*j+1] & 4) == 0) {
                v[i*M+j] = 0;
            } else {
                v[i*M+j] = 1;
            }
        }
    }
    
    //    std::cout<<"Vertices:"<<std::endl;
    //    for(int i=0; i<M; ++i) {
    //        for(int j=0; j<M; ++j) {
    //            std::cout<<v[i*M+j]<<" ";
    //        }
    //        std::cout<<std::endl;
    //    }
    
    return v;
}

domain TriangleDimerTiler::TilingToDomain(tiling &t) {
    // From tile state create domain
    int N = sqrt(t.size());
    int M = N/2+1;
    
    domain d((M-1)*2*(M-1),0);
    std::vector<int> v=TriangleDimerTiler::TilingToVertices(t);
    
    for(int i=0; i<M-1; ++i) {
        for(int j=0; j<2*(M-1); ++j) {
            if(j%2 == 0) {
                if(v[i*M+j/2] == 1 && v[i*M+j/2+1] == 1 && v[(i+1)*M+j/2] == 1) {
                    d[i*2*(M-1)+j] = 1;
                }
            } else {
                if(v[i*M+j/2+1] == 1 && v[(i+1)*M+j/2] == 1 && v[(i+1)*M+j/2+1] == 1) {
                    d[i*2*(M-1)+j] = 1;
                }
            }
        }
    }
    
//    std::cout<<"Domain:"<<std::endl;
//    for(int i=0; i<M-1; ++i) {
//        for(int j=0; j<2*(M-1); ++j) {
//            std::cout<<d[i*2*(M-1)+j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
    
    return d;
}

void TriangleDimerTiler::DomainToSVG(domain &d, std::string filename) {
    std::ofstream outputFile(filename.c_str());
    
    int M = sqrt(d.size()/2);
    
    outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"433.013\" width=\"500\" viewBox=\"0 0 "<<3/2.0*M<<" "<<3/2.0*M<<"\" preserveAspectRatio=\"none\">\n";
    outputFile << "<defs>\n";
    outputFile << "<g id=\"w\">  <polygon points = \"0,1 .5,0 1,1\" fill=\"paleturquoise\"/> </g>\n"; //upward triangle
    outputFile << "<g id=\"b\">  <polygon points = \"0,0 1,0 .5,1\" fill=\"deepskyblue\"/> </g>\n"; //downward triangle
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

void TriangleDimerTiler::DimerToSVG(tiling &t, std::string filename) {
	domain d = TilingToDomain(t);
    TriangleDimerTiler::DimerToSVG(t, d, filename);
}

void TriangleDimerTiler::TilingToSVG(tiling &t, std::string filename) {
	std::ofstream outputFile(filename.c_str());
    int N = sqrt(t.size());
    int M = N/2; // is M always N/2?

    outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"433.013\" width=\"500\" viewBox=\"0 0 "<<3/2.0*M<<" "<<3/2.0*M<<"\" preserveAspectRatio=\"none\">\n";
    outputFile << "<defs>\n";
    outputFile << "<g id=\"dR\">  <polygon points = \"1,-0.6666666667 0.5,-0.3333333333 0.5,0.3333333333 0,0.6666666667 0,1.333333333 0.5,1.666666667 1,1.333333333 1,0.6666666667 1.5,0.3333333333 1.5,-0.3333333333\" fill=\"midnightblue\" stroke=\"black\" stroke-width=\".05\"/></g>\n";
    outputFile << "<g id=\"dL\">  <polygon points = \"0,-0.6666666667 0.5,-0.3333333333 0.5,0.3333333333 1,0.6666666667 1,1.333333333 0.5,1.666666667 0,1.333333333 0,0.6666666667 -0.5,0.3333333333 -0.5,-0.3333333333\" fill=\"lightsteelblue\" stroke=\"black\" stroke-width=\".05\"/></g>\n";
    outputFile << "<g id=\"dH\">  <polygon points = \"0,-0.6666666667 0.5,-0.3333333333 1,-0.6666666667 1.5,-0.3333333333 1.5,0.3333333333 1,0.6666666667 0.5,0.3333333333 0,0.6666666667 -0.5,0.3333333333 -0.5,-0.3333333333\" fill=\"slategrey\" stroke=\"black\" stroke-width=\".05\"/></g>\n";
    outputFile << "</defs>\n";
    
    for(int i=0; i<M-1; ++i) {
        for(int j=0; j<2*M-1; ++j) {
            if(j%2 == 1) {
                if((t[2*i*N+j] & 4) == 4) {
                    outputFile<<"<use xlink:href = \"#dR\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if((t[2*i*N+j] & 8) == 8) {
                    outputFile<<"<use xlink:href = \"#dL\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if((t[(2*i+1)*N+j] & 1) == 1) {
                    outputFile<<"<use xlink:href = \"#dH\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
            }
        }
    }
    
    outputFile<< "</svg>";
    outputFile.close();
}


void TriangleDimerTiler::DimerToSVG(tiling &t, domain &d, std::string filename) {
    std::ofstream outputFile(filename.c_str());

    int N = sqrt(t.size());
    int M = sqrt(d.size()/2);

    outputFile << "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink=\"http://www.w3.org/1999/xlink\" height=\"433.013\" width=\"500\" viewBox=\"0 0 "<<3/2.0*M<<" "<<3/2.0*M<<"\" preserveAspectRatio=\"none\">\n";

    outputFile << "<defs>\n";
    outputFile << "<g id=\"dR\">  <polyline points= \".5,1 1,0\" stroke=\"darkblue\" stroke-width=\".05\"/> </g>\n"; // dimer SW to NE
    outputFile << "<g id=\"dL\">  <polyline points= \".5,1 0,0\" stroke=\"darkred\" stroke-width=\".05\"/> </g>\n"; // dimer NW to SE
    outputFile << "<g id=\"dH\">  <polyline points= \"1,0 0,0\" stroke=\"darkgreen\" stroke-width=\".05\"/> </g>\n"; // dimer horizontal
    outputFile << "<g id=\"eR\">  <polyline points= \".5,1 1,0\" stroke=\"black\" stroke-width=\".001\"/> </g>\n"; // edge SW to NE
    outputFile << "<g id=\"eL\">  <polyline points= \".5,1 0,0\" stroke=\"black\" stroke-width=\".001\"/> </g>\n"; // edge NW to SE
    outputFile << "<g id=\"eH\">  <polyline points= \"1,0 0,0\" stroke=\"black\" stroke-width=\".001\"/> </g>\n"; // edge horizontal
    outputFile << "<g id=\"v1\">  <ellipse cx=\".5\" cy=\"0\" rx=\".1\" ry=\".1\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/> </g> \n"; // vertex
    outputFile << "<g id=\"v2\">  <ellipse cx=\"1.5\" cy=\"0\" rx=\".1\" ry=\".1\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/> </g> \n"; // vertex
    outputFile << "<g id=\"v3\">  <ellipse cx=\"1\" cy=\"1\" rx=\".1\" ry=\".1\" stroke=\"black\" stroke-width=\".02\" fill=\"black\"/> </g> \n"; // vertex
    outputFile << "</defs>\n";

    for(int i=0; i<M-1; ++i) {
        for(int j=0; j<2*M-1; ++j) {
            if(j%2 == 1) {
                if((t[2*i*N+j] & 4) == 4) {
                    outputFile<<"<use xlink:href = \"#dR\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                } else if(d[i*2*M+j-1] == 1){
                    outputFile<<"<use xlink:href = \"#eR\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if((t[2*i*N+j] & 8) == 8) {
                    outputFile<<"<use xlink:href = \"#dL\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                } else if(d[i*2*M+j-1] == 1){
                    outputFile<<"<use xlink:href = \"#eL\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if((t[(2*i+1)*N+j] & 1) == 1) {
                    outputFile<<"<use xlink:href = \"#dH\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                } else if(d[i*2*M+j-1] == 1 || d[(i-1)*2*M+j] == 1) {
                    outputFile<<"<use xlink:href = \"#eH\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if((t[2*i*N+j] & 4) == 0 && d[i*2*M+j] == 1) {
                    outputFile<<"<use xlink:href = \"#eR\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if((t[2*i*N+j] & 8) == 0 && d[i*2*M+j-2] == 1 && j>1) {
                    outputFile<<"<use xlink:href = \"#eL\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
            }
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 2; j < 2*M; j++) {
            if (j%2 == 0) {
                // draw vertices. There is some redundancy here that could probably be fixed.
                if (d[i*2*M+j] == 1 || d[i*2*M+j-1] == 1) {
                    outputFile<<"<use xlink:href = \"#v1\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if (d[i*2*M+j] == 1 || d[i*2*M+j+1] == 1) {
                    outputFile<<"<use xlink:href = \"#v2\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
                if (d[i*2*M+j] == 1 || d[i*2*M+j+1] == 1 || d[i*2*M+j-1] == 1) {
                    outputFile<<"<use xlink:href = \"#v3\" x = \""<<(.5*j+.5*i)<<"\" y = \""<<i<<"\"/>\n";
                }
            }
        }
    }

    outputFile<< "</svg>";
    outputFile.close();

}




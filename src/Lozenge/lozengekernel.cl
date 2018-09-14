#include "./src/TinyMT/tinymt32.clh"

__kernel void RotateTiles(__global tinymt32wp_t * d_status, __global char* tiling, const int N) // fix all things with N
{
    // Attempts to rotate all tilings of a given color. Color determined by which tiling array is given as input.
    // Tilings are stored on vertices of hexagonal lattice. There is an indicator on each adjacent edge which is nonzero if a lozenge (or equivalently a dimer) crosses the edge. The value of the indicator on each edge is:
    //        1   2
    //     4 __\./__ 8
    //         / \
    //       16   32
    
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    tinymt32wp_t tiny;
    tinymt32_status_read(&tiny, d_status);
    float rd = tinymt32_single01(&tiny);
    tinymt32_status_write(d_status, &tiny);
    
    if ( i < N && j < N/3) {
        if (rd < .4) {
            if (tiling[i*(N/3)+j] == 25) tiling[i*(N/3)+j] = 38;
        } else if ( rd < .8 ) {
            if (tiling[i*(N/3)+j] == 38) tiling[i*(N/3)+j] = 25;
        }
//        if (tiling[i*(N/3)+j] == 25) {
//            tiling[i*(N/3)+j] = 38;
//        } else if (tiling[i*(N/3)+j] == 38) {
//            tiling[i*(N/3)+j] = 25;
//        }
        
    }
    
}

__kernel void UpdateTiles(__global char* tiling1, __global char* tiling2,__global char* tiling3, const int N, const int t)
{
    // Updates tilings, given the state of the flipped tilings (tiling1). Updates UR, L, and DR.
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if (i < N-1 && j < (N/3)-1 && i > 0 && j > 0) { // because we padded the arrays we can ignore the boundary
        int p2 = select(0,1,i%3 == (2*t+1)%3);
        int p4 = select(0,1,i%3 == (2*t+2)%3);
        int p8 = select(0,1,i%3 == (2*t)%3);
        int p16 = select(0,1,i%3 == (2*t+1)%3);
        
        tiling2[i*(N/3)+j] &= ~((tiling2[i*(N/3)+j] & 2) + (tiling2[i*(N/3)+j] & 4) + (tiling2[i*(N/3)+j] & 32)); //zero out appropriate bits
        tiling2[i*(N/3)+j] += (tiling1[(i-1)*(N/3)+j+p2] & 16)/8 + (tiling1[i*(N/3)+j-p4] & 8)/2 + (tiling1[(i+1)*(N/3)+j] & 1)*32; //update
        tiling3[i*(N/3)+j] &= ~((tiling3[i*(N/3)+j] & 1) + (tiling3[i*(N/3)+j] & 8) + (tiling3[i*(N/3)+j] & 16)); //zero out appropriate bits
        tiling3[i*(N/3)+j] += (tiling1[(i-1)*(N/3)+j] & 32)/32 + (tiling1[i*(N/3)+j+p8] & 4)*2 + (tiling1[(i+1)*(N/3)+j-p16] & 2)*8; //update
    }
}

__kernel void InitTinyMT(__global tinymt32wp_t * d_status, uint seed)
{
    tinymt32wp_t tiny;
    const size_t id = get_global_id(0);
    tinymt32_status_read(&tiny, d_status);
    tinymt32_init(&tiny, seed+id);
    tinymt32_status_write(d_status, &tiny);
}

#include "./src/TinyMT/tinymt32.clh"

// Tiling is stord on the vertices of the square lattice. There is an indicator on each adjacent edge which is nonzero if a domino (or equivalently a dimer) crosses the edge. The value of the indicator on each edge is:
//      1|
// 8 ____|____ 4
//       |
//      2|

// Attempts to rotate all tilings of a given color. Color determined by which tiling array is given as input.

#define cs .9f //cluster size
#define w 4.0f

 __kernel void RotateTiles(__global tinymt32wp_t * d_status, __global char* tiling, const int N, const int t)
{
    int i = get_global_id(0); int j = get_global_id(1);
    
    tinymt32wp_t tiny;
	tinymt32_status_read(&tiny, d_status);
  	float rd = tinymt32_single01(&tiny); 
 	tinymt32_status_write(d_status, &tiny);  
    
    float wr = select(select(1.0f/w, w , i%2 == 0), 1.0f, t == 0);
  
    if ( rd < cs && i < N && j < (N/2) ) {
    	rd /= cs;
        if ( tiling[i*(N/2)+j] == 3 && rd < wr) { tiling[i*(N/2)+j] = 12; }
        else if ( tiling[i*(N/2)+j]==12 && rd < 1.0/wr) { tiling[i*(N/2)+j] = 3; }
    }    
}

__kernel void UpdateTiles(__global char* tiling, __global char* reftiling, const int N, const int t)
{
    // Updates tilings, given the state of the adjacent tilings (reftiling)
    int i = get_global_id(0);
    int j = get_global_id(1);
     
    if (i < N-1 && j < (N/2)-1 && i>0 && j>0 ) {
        tiling[i*(N/2)+j] = (reftiling[(i-1)*(N/2)+j]&2)/2
        + 2*(reftiling[(i+1)*(N/2)+j]&1) 
        +(reftiling[i*(N/2)+j-(i+t+1)%2]&8)/2 
        + 2*(reftiling[i*(N/2)+j+(i+t)%2]&4);
    }
}

__kernel void InitTinyMT(__global tinymt32wp_t * d_status, uint seed)
{
    tinymt32wp_t tiny;
    const size_t id = get_global_id(0);
    tinymt32_status_read(&tiny, d_status);
    tinymt32_init(&tiny, seed);
    tinymt32_status_write(d_status, &tiny);
}

#include "./src/TinyMT/tinymt32.clh"

// Tiling is stord on the vertices of the square lattice. There is an indicator on each adjacent edge which is nonzero if a domino (or equivalently a dimer) crosses the edge. The value of the indicator on each edge is:
//      1|
// 8 ____|____ 4
//       |
//      2|

// Attempts to rotate all tilings of a given color. Color determined by which tiling array is given as input.

#define xo 48.0f
#define xe 0.5f
#define yo 16.0f
#define ye 0.125f

#define cs .9f //cluster size

 __kernel void RotateTiles(__global tinymt32wp_t * d_status, __global char* tiling, const int N, const int t)
{
    
    // the position in the tiling is:
    // i_t = i
    // j_t = 2*j + (t + i) % 2 ??
    
    int i = get_global_id(0); int j = get_global_id(1);
    
    tinymt32wp_t tiny;
  
	tinymt32_status_read(&tiny, d_status);
  	float rd = tinymt32_single01(&tiny); 
 	tinymt32_status_write(d_status, &tiny);  
    
    int pm = (1 + i + 2*j + (t+i) % 2 ) % 4;
   
    //float wr = select( select( select( xo/ye , ye/xe, pm == 2), xe/yo , pm == 1 ) , yo/xo , pm == 0);
    float wr = .9f;
      	
    wr = select(1.0f/wr, wr, t == 0);
  
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

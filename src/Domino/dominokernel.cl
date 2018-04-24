#include "./src/TinyMT/tinymt32.clh"


 __kernel void RotateTiles(__global tinymt32wp_t * d_status, __global char* tiling, const int N, const int t)
{
    int i = get_global_id(0)+1;
    int j = get_global_id(1)+1;
    
    tinymt32wp_t tiny;
	tinymt32_status_read(&tiny, d_status);
  	float rd = tinymt32_single01(&tiny); 
 	tinymt32_status_write(d_status, &tiny); 
    
    if ( rd < .8 ) {
    	if ( tiling[i*(N/2)+j] == 3 ) { tiling[i*(N/2)+j] = 12; }
    	else if ( tiling[i*(N/2)+j]==12 ) { tiling[i*(N/2)+j] = 3; }
    }
}


// t is the parity of the tiles being updated, see how this kernel is called in the RandomWalk method
__kernel void UpdateTiles(__global char* tiling, __global char* reftiling, const int N, const int t)
{
    int i = get_global_id(0)+1;
    int j = get_global_id(1)+1;
    
    tiling[i*(N/2)+j] = (reftiling[(i-1)*(N/2)+j]&2)/2
    			    	+ 2*(reftiling[(i+1)*(N/2)+j]&1) 
        				+ (reftiling[i*(N/2)+j-(i+t+1)%2]&8)/2 
        				+ 2*(reftiling[i*(N/2)+j+(i+t)%2]&4);
}

__kernel void InitTinyMT(__global tinymt32wp_t * d_status, uint seed)
{
    tinymt32wp_t tiny;
    const size_t id = get_global_id(0);
    tinymt32_status_read(&tiny, d_status);
    tinymt32_init(&tiny, seed+id);
    tinymt32_status_write(d_status, &tiny);
}

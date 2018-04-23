#include "./src/TinyMT/tinymt32.clh"

// Tiling is stored on the vertices of the square lattice. There is an indicator on each adjacent edge which is nonzero if a domino (or equivalently a dimer) crosses the edge. The value of the indicator on each edge is:
//      1|
// 8 ____|____ 4
//       |
//      2|

// Attempts to rotate all tilings of a given color. Color determined by which tiling array is given as input.

__kernel void RotateTiles(__global tinymt32wp_t * d_status, __global char* tiling, const int N, const int t)
{
	int i = get_global_id(0)+1;
	int j = get_global_id(1)+1;

	
	float rd = 0;

		tinymt32wp_t tiny;
		tinymt32_status_read(&tiny, d_status);
		rd = tinymt32_single01(&tiny);
		tinymt32_status_write(d_status, &tiny);
	
	
		if(rd < 0.5) {
			if (tiling[i*(N/2)+j] == 3) { tiling[i*(N/2)+j] = 12; }
		} else {
			if (tiling[i*(N/2)+j]==12) { tiling[i*(N/2)+j] = 3; }
		}
	
}

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

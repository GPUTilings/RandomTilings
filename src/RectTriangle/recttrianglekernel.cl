#include "./src/TinyMT/tinymt32.clh"

#define wt1 1.00000
#define wt2 1.00000
#define wt3 1.00000
#define wm1 1.00000
#define wm2 1.00000
#define wm3 1.00000
#define wr1 1.00000
#define wr2 1.00000
#define wr3 1.00000

__constant float weights[16] = {0.0, wt1, wt2, wt3, wr1, wr2, wr3, wr1, wr2, wr3, wm1, wm2, wm3, wm1, wm2, wm3};

float getWeightRatio(int a,int b);

// Tiling is stored on vertices of the triangular lattice. Each adjacent face if given a integer value in [0,15] based on how it is covered by the tiling. These values are stored as a six digit hexidecimal int as follows:
//     ___
//   /\   /\         d4
//  /__\./__\ -> d5      d3 ->  d5*16^5 + d4*16^4 + d3*16^3 + d2*16^2 + d1*16^1 + d0*16^0
//  \  / \  /    d2      d0
//   \/___\/         d1
//

__kernel void flipTiles(__global tinymt32wp_t * d_status, __global int* tiling, const int N)
{
    // Lots of 'if' statements, is there a better way?
    // Attempts to flip all tilings of a given color. Color determined by which tiling array is given as input.
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    tinymt32wp_t tiny;
    tinymt32_status_read(&tiny, d_status);
    float rd = tinymt32_single01(&tiny);
    tinymt32_status_write(d_status, &tiny);
    float bs = 0.9;
    if ( i < N && j < N/3 && rd < bs) {
        rd /= bs;
        int hexType = tiling[i*(N/3)+j];
        int hex1; int hex2; int newType = hexType;
        float sc1, sc2, sc;
        sc1 = select(0.75f,0.5f,rd < 0.75f); sc2 = select(sc1,0.25f,rd < 0.5f); sc = select(sc2,0.0f,rd < 0.25f);
        if (hexType == 0x122133) { /* all triangles (1) */
            // select(a,b,c), a=value  if false, b=value if true, c=condition
            hex1 = select(0x331221,0x8ebbe8,rd < 0.75); // if 0.5 < rd < 0.75 flip to 0x8ebbe8
            hex2 = select(hex1,0xfc99cf,rd < 0.5); // if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = select(hex2,0xa7dd7a,rd < 0.25); // if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x331221
        } else if (hexType == 0x331221) { /* all triangles (3) */
            hex1 = select(0x122133,0x8ebbe8,rd < 0.75); // if 0.25 < rd < 0.75 flip to 0x8ebbe8
            hex2 = select(hex1,0xfc99cf,rd < 0.5); // if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = select(hex2,0xa7dd7a,rd < 0.25); // if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x122133
        } else if (hexType == 0x8ebbe8) { /* single rect (b) */
            hex1 = select(0x331221,0x122133,rd < 0.75); // if 0.5 < rd < 0.75 flip to 0x122133
            hex2 = select(hex1,0xfc99cf,rd < 0.5); // if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = select(hex2,0xa7dd7a,rd < 0.25); // if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x331221
        } else if (hexType == 0xfc99cf) { /* single rect (c) */
            hex1 = select(0x331221,0x8ebbe8,rd < 0.75); // if 0.5 < rd < 0.75 flip to 0x8ebbe8
            hex2 = select(hex1,0x122133,rd < 0.5); // if 0.25 < rd < 0.5 flip to 0x122133
            newType = select(hex2,0xa7dd7a,rd < 0.25); // if 0 < rd < 0.25 flip to 0xa7dd7a, else 0x331221
        } else if (hexType == 0xa7dd7a) { /* single rect (a) */
            hex1 = select(0x331221,0x8ebbe8,rd < 0.75); // if 0.5 < rd < 0.75 flip to 0x8ebbe8
            hex2 = select(hex1,0xfc99cf,rd < 0.5); // if 0.25 < rd < 0.5 flip to 0xfc99cf
            newType = select(hex2,0x122133,rd < 0.25); // if 0 < rd < 0.25 flip to 0x122133, else 0x331221
        }
        
        else if (hexType == 0x47d47a) { /* rect + left rect (a) */
            newType = 0xd22a33;
            sc = 1.0;
        } else if (hexType == 0xd22a33) { /* triangles + left rect (a) */
            newType = 0x47d47a;
            sc = 1.0;
        }
        
        else if (hexType == 0xa74d74) { /* rect + right rect (a) */
            newType = 0x33a22d;
            sc = 1.0;
        } else if (hexType == 0x33a22d) { /* triangles + right rect (a) */
            newType = 0xa74d74;
            sc = 1.0;
        }
        
        else if (hexType == 0x8eb558 ) { /* rect + left rect (b) */
            newType = 0x331eb1;
            sc = 1.0;
        } else if (hexType == 0x331eb1 ) { /* triangles + left rect (b) */
            newType = 0x8eb558;
            sc = 1.0;
        }
        
        else if (hexType == 0x855be8 ) { /* rect + right rect (b) */
            newType = 0x1be133;
            sc = 1.0;
        } else if (hexType == 0x1be133 ) { /* triangles + right rect (b) */
            newType = 0x855be8;
            sc = 1.0;
        }
        
        else if (hexType == 0x6699cf ) { /* rect + left rect (c) */
            newType = 0xcf1221;
            sc = 1.0;
        } else if (hexType == 0xcf1221 ) { /* triangles + left rect (c) */
            newType = 0x6699cf;
            sc = 1.0;
        }
        
        else if (hexType == 0xfc9966 ) { /* rect + right rect (c) */
            newType = 0x1221fc;
            sc = 1.0;
        } else if (hexType == 0x1221fc ) { /* triangles + right rect (c) */
            newType = 0xfc9966;
            sc = 1.0;
        }
        
        float weightratio = getWeightRatio(hexType,newType);
        rd = select((rd-sc)*4,rd,sc==1.0); //rescale rd
        tiling[i*(N/3)+j] = select(hexType,newType, weightratio >= rd);
    }
}

__kernel void updateTiles1(__global int* tiling1, __global int* tiling2, const int N, const int t)
{
    // nned to fix indexing
    // Updates tilings, given the state of the flipped tilings (tiling1).
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i < N-1 && j < (N/3)-1 && i>0 && j>0) { // because we padded the arrays we can ignore the boundary
        int p1 = select(0,1,i%3==t);
        int p2 = select(0,1,i%3==(t+1)%3);
        tiling2[i*(N/3)+j] = (tiling1[(i-1)*(N/3)+j]&0x0000f0)*0x010000 + (tiling1[(i-1)*(N/3)+j]&0x00000f)*0x010000 + (tiling1[i*(N/3)+j+p1]&0xf00000)/0x000100 + (tiling1[(i+1)*(N/3)+j-p2]&0x0f0000)/0x000100 + (tiling1[(i+1)*(N/3)+j-p2]&0x00f000)/0x000100 + (tiling1[i*(N/3)+j+p1]&0x000f00)/0x000100;
    }
}

__kernel void updateTiles2(__global int* tiling1, __global int* tiling2, const int N, const int t)
{
    // need to fix indexing
    // Updates tilings, given the state of the flipped tilings (tiling1).
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i < N-1 && j < (N/3)-1 && i>0 && j>0) { // because we padded the arrays we can ignore the boundary
        int p1 = select(0,1,i%3==(t+2)%3);
        int p2 = select(0,1,i%3==(t+1)%3);
        tiling2[i*(N/3)+j] = (tiling1[i*(N/3)+j-p1]&0x00f000)*0x000100 + (tiling1[(i-1)*(N/3)+j+p2]&0x000f00)*0x000100 + (tiling1[(i-1)*(N/3)+j+p2]&0x0000f0)*0x000100 + (tiling1[i*(N/3)+j-p1]&0x00000f)*0x000100 + (tiling1[(i+1)*(N/3)+j]&0xf00000)/0x010000 + (tiling1[(i+1)*(N/3)+j]&0x0f0000)/0x010000;
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

float getWeightRatio(int a, int b)
{
    return ((weights[(b&0xf00000)/0x100000]*weights[(b&0x0f0000)/0x010000]*weights[(b&0x00f000)/0x001000]*weights[(b&0x000f00)/0x000100]*weights[(b&0x0000f0)/0x000010]*weights[(b&0x00000f)/0x000001]) / (weights[(a&0xf00000)/0x100000]*weights[(a&0x0f0000)/0x010000]*weights[(a&0x00f000)/0x001000]*weights[(a&0x000f00)/0x000100]*weights[(a&0x0000f0)/0x000010]*weights[(a&0x00000f)/0x000001]));
}


#include "./src/TinyMT/tinymt32.clh"


__kernel void RotateLozenges(__global tinymt32wp_t * d_status, __global int* tiling, const int N, const int t, const int c)
{
    // Attempts a Lozenge type flip
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    tinymt32wp_t tiny;
    tinymt32_status_read(&tiny, d_status);
    float rd = tinymt32_single01(&tiny);
    tinymt32_status_write(d_status, &tiny);
    
     if ( rd < .5 && i < N && j < N && ((1-(t&1))*i + ((1+t)/2) * j)%2 == c) {
         if (tiling[i*N+j] == 5) {
             tiling[i*N+j] = 10;
         } else if (tiling[i*N+j] == 10) {
             tiling[i*N+j] = 5;
         }
     }
}

__kernel void UpdateLozengesFlipped(__global int* tiling, const int N, const int t, const int c)
{
    // Updates the state of all elements of the tiling the same as those just flipped. t is the orientation of those just flipped (0=H, 1=L, 2=R), c indicates which subset of a given orientation tries to flip.
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i < N-1 && j < N-1 && i>0 && j>0 && ((1-(t&1))*i + ((1+t)/2) * j)%2 != c) {
        tiling[i*N+j] = (tiling[(i-1)*N+j+(t&1)] & 4)/4 + (tiling[(i-(2-t)/2)*N+j+1] & 8)/4 + (tiling[(i+1)*N+j-(t&1)] & 1)*4 + (tiling[(i+(2-t)/2)*N+j-1] & 2)*4;
    }
}

__kernel void UpdateLozenges0(__global int* tiling1, __global int* tiling2, __global int* tiling3, const int N)
{
    // After flipping and updating tilings of type 0 (Horizontal), we now update tilings of type 1 (Left) and 2 (Right).
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {
        tiling2[i*N+j] &= ~10;
        tiling2[i*N+j] |= (tiling1[(i+1)*N+j] & 1)*2 + (tiling1[(i+1)*N+j-1] & 1)*8;
        tiling3[i*N+j] &= ~10;
        tiling3[i*N+j] |= (tiling1[(i+1)*N+j] & 2) + (tiling1[i*N+j] &  8);
    }
}

__kernel void UpdateLozenges1(__global int* tiling1, __global int* tiling2, __global int* tiling3, const int N)
{
    // After flipping and updating tilings of type 1 (Left), we now update tilings of type 2 (Right) and 0 (Horizontal).
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {
        tiling2[i*N+j] &= ~5;
        tiling2[i*N+j] |= (tiling1[i*N+j] & 1) + (tiling1[(i+1)*N+j] & 1)*4;
        tiling3[i*N+j] &= ~5;
        tiling3[i*N+j] |= (tiling1[(i-1)*N+j] & 2)/2 + (tiling1[i*N+j] & 2)*2;
    }
}

__kernel void UpdateLozenges2(__global int* tiling1, __global int* tiling2, __global int* tiling3, const int N)
{
    // After flipping and updating tilings of type 2 (Right), we now update tilings of type 0 (Horizontal) and 1 (Left).
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {
        tiling2[i*N+j] &= ~10;
        tiling2[i*N+j] |= (tiling1[(i-1)*N+j] & 2) + (tiling1[i*N+j-1] & 2)*4;
        tiling3[i*N+j] &= ~5;
        tiling3[i*N+j] |= (tiling1[i*N+j] & 1) + (tiling1[i*N+j-1] & 4);
    }
}

__kernel void UpdateLozenges(__global int* tiling1, __global int* tiling2, __global int* tiling3, const int N, const int t)
{
    // Not used.
    // Updates tilings, given the state of the adjacent tilings
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if (i < N-1 && j < N-1 && i>0 && j>0 ) {
        tiling2[i*N+j] &= ~(2-(t&1) + 8-4*(t&1));
        tiling2[i*N+j] |= (tiling1[(i+1-t)*N+j] & 1+t/2)*1 + (tiling1[(i+1-t/2)*N+j-1+(t&1)] & 1+t/2)*(4+4*((2-t)/2));
        tiling3[i*N+j] &= ~(1+((2-t)/2) + 4+4*((2-t)/2));
        tiling3[i*N+j] |= (tiling1[(i-(t&1)+(2-t)/2)*N+j] & 2-t/2)/(1+(t&1)) + (tiling1[i*N+j-t/2] & 4*(t/2) + 2*(t&1) + 8*((2-t)/2))*(1+(t&1));
    }
}

__kernel void UpdateTriangleUFromLozenges(__global int* tilingH, __global int* tilingL, __global int* tilingU, const int N)
{
    // Update Up triangles from Lozenges
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        int L1 = select(0,1,(tilingL[i*N+j] & 1) == 1);
        int L2 = select(0,2,(tilingH[i*N+j] & 8) == 8);
        int L3 = select(0,3,(tilingH[(i+1)*N+j] & 1) == 1);
        int R1 = select(0,1,(tilingL[i*N+j+1] & 1) == 1);
        int R2 = select(0,2,(tilingH[(i+1)*N+j] & 2) == 2);
        int R3 = select(0,3,(tilingL[i*N+j+1] & 2) == 2);
        int V1 = select(0,1,(tilingL[i*N+j+1] & 4) == 4);
        int V2 = select(0,2,(tilingH[(i+1)*N+j] & 8) == 8);
        int V3 = select(0,3,(tilingH[(i+1)*N+j] & 4) == 4);
        
        tilingU[i*N+j] = L1+L2+L3 + 4*(R1+R2+R3) + 16*(V1+V2+V3);
    }
}

__kernel void UpdateButterflysHFromLozenge(__global int* tilingBH, __global int* tilingLH, __global int* tilingLL, __global int* tilingLR, const int N)
{
    // Update horizontal butterflys from lozenges
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {
        int LH = tilingLH[i*N+j];
        int LR1 = tilingLR[(i-1)*N+j];
        int LL1 = tilingLL[(i-1)*N+j+1];
        int LR2 = tilingLR[i*N+j];
        int LL2 = tilingLL[i*N+j];
        
        int c = (LL1 & 4)/4;
        int t1 = (LR1 & 8)/8 + 2*(LR1 & 1) + 3*(LH & 1);
        int t4 = (LL1 & 1) + (LL1 & 2) + ((LH & 2)/2)*3;
        int t16 = (LR2 & 2)/2 + (LR2 & 4)/2 + ((LH & 4)/4)*3;
        int t64 = (LL2 & 4)/4 + (LL2 & 8)/4 + ((LH & 8)/8)*3;
        
        tilingBH[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
    }
}

__kernel void UpdateButterflysLFromLozenge(__global int* tilingBL, __global int* tilingLH, __global int* tilingLL, __global int* tilingLR, const int N)
{
    // Update left butterflys from lozenges
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {
        int LL = tilingLL[i*N+j];
        int LH1 = tilingLH[i*N+j];
        int LR1 = tilingLR[i*N+j];
        int LH2 = tilingLH[(i+1)*N+j-1];
        int LR2 = tilingLR[i*N+j-1];
        
        int c = (LR1 & 8)/8;
        int t1 = (LH1 & 1) + (LH1 & 2) + 3*(LL & 1);
        int t4 = (LR1 & 2)/2 + (LR1 & 4)/2 + ((LL & 2)/2)*3;
        int t16 = (LH2 & 4)/4 + (LH2 & 8)/4 + ((LL & 4)/4)*3;
        int t64 = (LR2 & 8)/8 + 2*(LR2 & 1) + ((LL & 8)/8)*3;
        
        tilingBL[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
    }
}

__kernel void UpdateButterflysRFromLozenge(__global int* tilingBR, __global int* tilingLH, __global int* tilingLL, __global int* tilingLR, const int N)
{
    // Update right butterflys from lozenges
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {
        int LR = tilingLR[i*N+j];
        int LH1 = tilingLH[i*N+j];
        int LL1 = tilingLL[i*N+j+1];
        int LH2 = tilingLH[(i+1)*N+j];
        int LL2 = tilingLL[i*N+j];
        
        int c = (LL1 & 8)/8;
        int t1 = (LH1 & 1) + (LH1 & 2) + 3*(LR & 1);
        int t4 = (LL1 & 1) + (LL1 & 2) + ((LR & 2)/2)*3;
        int t16 = (LH2 & 4)/4 + (LH2 & 8)/4 + ((LR & 4)/4)*3;
        int t64 = (LL2 & 4)/4 + (LL2 & 8)/4 + ((LR & 8)/8)*3;
        
        tilingBR[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
    }
    
}

__kernel void RotateTriangles(__global tinymt32wp_t * d_status, __global int* tiling, const int N, const int c)
{
    // Attempts a Triangle type flip.
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    tinymt32wp_t tiny;
    tinymt32_status_read(&tiny, d_status);
    float rd = tinymt32_single01(&tiny);
    tinymt32_status_write(d_status, &tiny);
    
    if ( rd < 0.5 && i < N && j < N && ((j-i)%3+3)%3 == c) {
        if (tiling[i*N+j] == 45) {
            tiling[i*N+j] = 54;
        } else if (tiling[i*N+j] == 54) {
            tiling[i*N+j] = 45;
        }
    }
    
    
}

__kernel void UpdateTrianglesFlipped0(__global int* tiling, const int N, const int c)
{
    // Update up triangles after up flips
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 && ((j-i)%3+3)%3 != c) {
        bool b = (((j-i)%3+3)%3 == (c+1)%3);
        int p = select(0, 1, b);
        int t1 = (4-3*p)*12;
        int m1 = (4-3*p)*4;
        int t4 = (15*p+1)*3;
        int m4 = (15*p+1);
        int t16 = (4-3*p)*3;
        int m16 = (4-3*p);
        
        tiling[i*N+j] = (tiling[(i-(1-p))*N+j-p] & t1)/m1 + ((tiling[(i-p)*N+j+1] & t4)/m4)*4 + ((tiling[(i+1)*N+j-1+p] & t16)/m16)*16;
    }
    
}

__kernel void UpdateTrianglesFlipped1(__global int* tiling, const int N, const int c)
{
    // Update down triangles after down flips
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 && ((j-i)%3+3)%3 != c) {
        bool b = (((j-i)%3+3)%3 == (c+1)%3);
        int p = select(0, 1, b);
        int t1 = (4-3*p)*12;
        int m1 = (4-3*p)*4;
        int t4 = (15*p+1)*3;
        int m4 = (15*p+1);
        int t16 = (4-3*p)*3;
        int m16 = (4-3*p);
        
        tiling[i*N+j] = (tiling[(i+1-p)*N+j-1] & t1)/m1 + ((tiling[(i+p)*N+j+1-p] & t4)/m4)*4 + ((tiling[(i-1)*N+j+p] & t16)/m16)*16;
    }
    
}

__kernel void UpdateTriangles(__global int* tiling1, __global int* tiling2, const int N, const int t)
{
    // Update down triangles form up triangle (t=0), or up triangles from down triangles (t=1)
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        int p = select(1,-1,t==0);
        int L1 = select(0,1,(tiling1[i*N+j-1+t] & 48) == 16);
        int L2 = select(0,2,(tiling1[i*N+j-1+t] & 3) == 3);
        int L3 = select(0,3,(tiling1[i*N+j-1+t] & 12) == 8);
        int R1 = select(0,1,(tiling1[i*N+j+t] & 48) == 16);
        int R2 = select(0,2,(tiling1[i*N+j+t] & 3) == 3);
        int R3 = select(0,3,(tiling1[i*N+j+t] & 12) == 8);
        int V1 = select(0,1,(tiling1[(i+p)*N+j] & 48) == 16);
        int V2 = select(0,2,(tiling1[(i+p)*N+j] & 3) == 3);
        int V3 = select(0,3,(tiling1[(i+p)*N+j] & 12) == 8);
        
        tiling2[i*N+j] = L1+L2+L3 + 4*(R1+R2+R3) + 16*(V1+V2+V3);
    }
}

__kernel void UpdateLozengeHFromTriangles(__global int* tilingU, __global int* tilingD, __global int* tilingH, const int N)
{
    // update horizontal lozenges from up triangles
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        int tD = tilingD[i*N+j];
        int tU = tilingU[(i-1)*N+j];
        int H1 = select(0,1, (tD & 48) == 32);
        int H2 = select(0,1, (tD & 48) == 48);
        int H4 = select(0,1, (tU & 48) == 48);
        int H8 = select(0,1, (tU & 48) == 32);
        tilingH[i*N+j] = H1+2*H2+4*H4+8*H8;
    }
}

__kernel void UpdateLozengeLFromTriangles(__global int* tilingU, __global int* tilingD, __global int* tilingL, const int N)
{
    // update left lozenges from up triangles
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        int tD = tilingD[i*N+j];
        int tU = tilingU[i*N+j-1];
        int L1 = select(0,1, (tU & 12) == 4);
        int L2 = select(0,1, (tU & 12) == 12);
        int L4 = select(0,1, (tD & 3) == 1);
        int L8 = select(0,1, (tD & 3) == 2);
        tilingL[i*N+j] = L1+2*L2+4*L4+8*L8;
    }
}

__kernel void UpdateLozengeRFromTriangles(__global int* tilingU, __global int* tilingD, __global int* tilingR, const int N)
{
    // update right lozenges from up triangles
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        int tD = tilingD[i*N+j];
        int tU = tilingU[i*N+j];
        int R1 = select(0,1, (tU & 3) == 1);
        int R2 = select(0,1, (tD & 12) == 12);
        int R4 = select(0,1, (tD & 12) == 4);
        int R8 = select(0,1, (tU & 3) == 2);
        tilingR[i*N+j] = R1+2*R2+4*R4+8*R8;
    }
}

__kernel void RotateButterflys(__global tinymt32wp_t * d_status, __global int* tiling, const int N, const int t, const int p1, const int p2)
{
    // Attempts a Butterfly type flip
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    tinymt32wp_t tiny;
    tinymt32_status_read(&tiny, d_status);
    float rd = tinymt32_single01(&tiny);
    tinymt32_status_write(d_status, &tiny);
    
    int b1 = (1-(t&1));
    int b2 = ((1+t)/2);
    int b3 = (t&1);
    
    if (rd < 0.5 && i < N && j < N  && ((b1*i + b2*j)%3) == p1 ) {
        if (tiling[i*N+j] == 170) {
            tiling[i*N+j] = select(170,85,((b1*j + b3*i)&1) == p2);
        } else if (tiling[i*N+j] == 85) {
            tiling[i*N+j] = select(85,170,((b1*j + b3*i)&1) == p2);
        }
    }
}

// Series of kernels for updating butterflys after butterfly flips. See triangledimer.cpp for a description.
__kernel void UpdateButterflysFlippedH1(__global int* tiling, const int N, const int p1, const int p2)
{
    // Update same slice
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((i%3) == p1 && (j&1) != p2) {
            int Bf1 = tiling[i*N+j-1];
            int Bf2 = tiling[i*N+j+1];
            
            int c = (tiling[i*N+j]&256)/256;
            int t1 = select(((Bf1 & 12)/4)%3  + 1,0,(Bf1 & 12)/4 == 0);
            int t4 = select(((Bf2 & 3)+1)%3  + 1,0,(Bf2 & 3) == 0);
            int t16 = select(((Bf2 & 192)/64)%3 + 1,0,(Bf2 & 192)/64 == 0);
            int t64 = select(((Bf1 & 48)/16 + 1)%3 + 1,0,(Bf1 & 48)/16 == 0);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
    
}

__kernel void UpdateButterflysFlippedH21(__global int* tiling, const int N, const int p1)
{
    // Update adjacent slice partially
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((i%3) == (p1+1)%3 ) { //up
            int Bf = tiling[i*N+j];
            int Bf1 = tiling[(i-1)*N+j];
            int Bf2 = tiling[(i-1)*N+j+1];
            
            int c = select(0,1,(Bf1 & 48)/16 == 2);
            int t1 = select((Bf1 & 192)/192 + ((Bf1 & 48)/48)*3,(Bf & 3),(Bf & 3) == 2);
            int t4 = select(((Bf2 & 192)/192)*3 + ((Bf2 & 48)/48)*2,(Bf&12)/4,(Bf & 12)/4 == 1);
            int t16 = (Bf & 48)/16;
            int t64 = (Bf & 192)/64;
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
}

__kernel void UpdateButterflysFlippedH22(__global int* tiling, const int N, const int p1)
{
    // Update adjacent slice
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((i%3) == (p1+2)%3 ) {
            int Bf = tiling[i*N+j];
            int s1 = select(0,1,(Bf & 3) != 2);
            int s4 = select(0,1,(Bf & 12)/4 != 1);
            int Bf1 = tiling[(i-1)*N+j];
            int Bf2 = tiling[(i-1)*N+j+1];
            int Bf3 = tiling[(i+1)*N+j];
            int Bf4 = tiling[(i+1)*N+j-1];
            
            int c = select(0,1,(Bf3 & 3) == 2);
            int t1 = select(s1*(Bf & 3),2,(Bf1 & 256) == 256);
            int t4 = select(s4*((Bf & 12)/4),1,(Bf2 & 256) == 256);
            int t16 = select(((Bf3 & 3)/3)*3 + (Bf3 & 12)/12,(Bf & 48)/16,(Bf & 48)/16 == 2);
            int t64 = select(((Bf4 & 3)/3)*2 + ((Bf4 & 12)/12)*3,(Bf & 192)/64,(Bf & 192)/64 == 1);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
}

__kernel void UpdateButterflysFlippedH23(__global int* tiling, const int N, const int p1)
{
    // Finish updating adjacent slice in H21
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((i%3) == (p1+1)%3 ) {
            int Bf = tiling[i*N+j];
            int s16 = select(0,1,(Bf & 48)/16 != 2);
            int s64 = select(0,1,(Bf & 192)/64 != 1);
            int Bf3 = tiling[(i+1)*N+j];
            int Bf4 = tiling[(i+1)*N+j-1];
            
            int c = (Bf & 256)/256;
            int t1 = (Bf & 3);
            int t4 = (Bf & 12)/4;
            int t16 = select(s16*((Bf & 48)/16),2,(Bf3 & 256) == 256);;
            int t64 = select(s64*((Bf & 192)/64),1,(Bf4 & 256) == 256);;
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
}

__kernel void UpdateButterflysFlippedL1(__global int* tiling, const int N, const int p1, const int p2)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((j%3) == p1 && (i&1) != p2)
        {
            int Bf1 = tiling[(i-1)*N+j];
            int Bf2 = tiling[(i+1)*N+j];
            int c = (tiling[i*N+j] & 256)/256;
            int t1 = select(((Bf1 & 12)/4)%3  + 1,0,(Bf1 & 12)/4 == 0);
            int t4 = select(((Bf2 & 3)+1)%3  + 1,0,(Bf2 & 3) == 0);
            int t16 = select(((Bf2 & 192)/64)%3 + 1,0,(Bf2 & 192)/64 == 0);
            int t64 = select(((Bf1 & 48)/16 + 1)%3 + 1,0,(Bf1 & 48)/16 == 0);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
    
}

__kernel void UpdateButterflysFlippedL21(__global int* tiling, const int N, const int p1)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((j%3) == (p1+2)%3) { //right
            int Bf = tiling[i*N+j];
            int Bf1 = tiling[(i-1)*N+j+1];
            int Bf2 = tiling[i*N+j+1];
            
            int c = select(0,1,(Bf1 & 48)/16 == 2);
            int t1 = select((Bf1 & 192)/192 + ((Bf1 & 48)/48)*3,(Bf & 3),(Bf & 3) == 2);
            int t4 = select(((Bf2 & 192)/192)*3 + ((Bf2 & 48)/48)*2,(Bf&12)/4,(Bf & 12)/4 == 1);
            int t16 = (Bf & 48)/16;
            int t64 = (Bf & 192)/64;
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
    
}

__kernel void UpdateButterflysFlippedL22(__global int* tiling, const int N, const int p1)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((j%3) == (p1+1)%3) { //left
            int Bf = tiling[i*N+j];
            int s1 = select(0,1,(Bf & 3) != 2);
            int s4 = select(0,1,(Bf & 12)/4 != 1);
            
            int Bf1 = tiling[(i-1)*N+j+1];
            int Bf2 = tiling[i*N+j+1];
            int Bf3 = tiling[(i+1)*N+j-1];
            int Bf4 = tiling[i*N+j-1];
            
            int c = select(0,1,(Bf3 & 3) == 2);
            int t1 = select(s1*(Bf & 3),2,(Bf1 & 256)/256 == 1);
            int t4 = select(s4*(Bf & 12)/4,1,(Bf2 & 256)/256 == 1);
            int t16 = select(((Bf3 & 3)/3)*3 + (Bf3 & 12)/12,(Bf & 48)/16,(Bf & 48)/16 == 2);
            int t64 = select(((Bf4 & 3)/3)*2 + ((Bf4 & 12)/12)*3,(Bf & 192)/64,(Bf & 192)/64 == 1);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
    
}

__kernel void UpdateButterflysFlippedL23(__global int* tiling, const int N, const int p1)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if((j%3) == (p1+2)%3) { //right
            int Bf = tiling[i*N+j];
            int s16 = select(0,1,(Bf & 48)/16 != 2);
            int s64 = select(0,1,(Bf & 192)/64 != 1);
            
            int Bf3 = tiling[(i+1)*N+j-1];
            int Bf4 = tiling[i*N+j-1];
            
            int c = (Bf & 256)/256;
            int t1 = (Bf & 3);
            int t4 = (Bf & 12)/4;
            int t16 = select(s16*(Bf & 48)/16,2,(Bf3 & 256)/256 == 1);
            int t64 = select(s64*(Bf & 192)/64,1,(Bf4 & 256)/256 == 1);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
    
}

__kernel void UpdateButterflysFlippedR1(__global int* tiling, const int N, const int p1, const int p2)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if(((i + j)%3) == p1 && (j&1) != p2) {
            int Bf1 = tiling[(i+1)*N+j-1];
            int Bf2 = tiling[(i-1)*N+j+1];
            
            int c = (tiling[i*N+j] & 256)/256;
            int t1 = select(((Bf2 & 192)/64 + 1)%3 + 1,0,(Bf2 & 192)/64 == 0);
            int t4 = select(((Bf2 & 48)/16)%3 + 1,0,(Bf2 & 48)/16 == 0);
            int t16 = select(((Bf1 & 12)/4 + 1)%3 + 1,0,(Bf1 & 12)/4 == 0);
            int t64 = select(((Bf1 & 3))%3+1,0,(Bf1 & 3) == 0);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
}

__kernel void UpdateButterflysFlippedR21(__global int* tiling, const int N, const int p1)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if(((i + j)%3) == (p1+2)%3) { //right
            int Bf = tiling[i*N+j];
            int Bf2 = tiling[i*N+j+1];
            int Bf3 = tiling[(i+1)*N+j];
            
            int c = select(0,1,(Bf3 & 3) == 1);
            int t1 = (Bf & 3);
            int t4 = select(((Bf2 & 3)/3) + ((Bf2 & 192)/192)*3,(Bf & 12)/4,(Bf & 12)/4 == 2);
            int t16 = select(((Bf3 & 3)/3)*3 + ((Bf3 & 192)/192)*2,(Bf & 48)/16,(Bf & 48)/16 == 1);
            int t64 = (Bf & 192)/64;
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
    
}

__kernel void UpdateButterflysFlippedR22(__global int* tiling, const int N, const int p1)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if(((i + j)%3) == (p1+1)%3) { //left
            int Bf = tiling[i*N+j];
            int s4 = select(0,1,(Bf & 12)/4 != 2);
            int s16 = select(0,1,(Bf & 48)/16 != 1);
            
            int Bf1 = tiling[(i-1)*N+j];
            int Bf2 = tiling[i*N+j+1];
            int Bf3 = tiling[(i+1)*N+j];
            int Bf4 = tiling[i*N+j-1];
            
            int c = select(0,1,(Bf1 & 48)/16 == 1);
            int t1 = select(((Bf1 & 48)/48)*3 + ((Bf1 & 12)/12)*2,(Bf & 3),(Bf & 3) == 1);
            int t4 = select(s4*(Bf & 12)/4,2,(Bf2 & 256)/256 == 1);
            int t16 = select(s16*(Bf & 48)/16,1,(Bf3 & 256)/256 == 1);
            int t64 = select(((Bf4 & 48)/48) + ((Bf4 & 12)/12)*3,(Bf & 192)/64,(Bf & 192)/64 == 2);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
    
}

__kernel void UpdateButterflysFlippedR23(__global int* tiling, const int N, const int p1)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0) {
        if(((i + j)%3) == (p1+2)%3) { //right
            int Bf = tiling[i*N+j];
            int s1 = select(0,1,(Bf & 3) != 1);
            int s64 = select(0,1,(Bf & 192)/64 != 2);
            
            int Bf1 = tiling[(i-1)*N+j];
            int Bf4 = tiling[i*N+j-1];
            
            int c = (Bf & 256)/256;
            int t1 = select(s1*(Bf & 3),1,(Bf1 & 256)/256 == 1);
            int t4 = (Bf & 12)/4;
            int t16 = (Bf & 48)/16;
            int t64 = select(s64*(Bf & 192)/64,2,(Bf4 & 256)/256 == 1);
            
            tiling[i*N+j] = t1 + 4*t4 + 16*t16 + 64*t64 + 256*c;
        }
    }
}


__kernel void UpdateLozengeFromButterflysH(__global int* tilingBH, __global int* tilingLH,__global int* tilingLL,__global int* tilingLR, const int N)
{
    // update lozenges from horizontal butterfly
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {
        int BF = tilingBH[i*N+j];
        int lh1 = select(0,1,(BF & 3)==3);
        int lh2 = select(0,1,(BF & 12)==12);
        int lh4 = select(0,1,(BF & 48)==48);
        int lh8 = select(0,1,(BF & 192)==192);
        tilingLH[i*N+j] = lh1+2*lh2+4*lh4+8*lh8;
        
        int ll1 = select(0,1,(BF & 256)==256);
        int ll2 = select(0,1,(BF & 48)==48);
        int ll4 = select(0,1,(BF & 192)==64);
        int ll8 = select(0,1,(BF & 192)==128);
        tilingLL[i*N+j] = ll1+2*ll2+4*ll4+8*ll8;
        
        int lr1 = select(0,1,(BF & 256)==256);
        int lr2 = select(0,1,(BF & 48)==16);
        int lr4 = select(0,1,(BF & 48)==32);
        int lr8 = select(0,1,(BF & 192)==192);
        tilingLR[i*N+j] = lr1+2*lr2+4*lr4+8*lr8;
    }
    
}


__kernel void UpdateLozengeFromButterflysL(__global int* tilingBL,__global int* tilingLH,__global int* tilingLL,__global int* tilingLR, const int N)
{
    // update lozenges from left butterfly
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {
        int BF = tilingBL[i*N+j];
        int ll1 = select(0,1,(BF & 3)==3);
        int ll2 = select(0,1,(BF & 12)==12);
        int ll4 = select(0,1,(BF & 48)==48);
        int ll8 = select(0,1,(BF & 192)==192);
        tilingLL[i*N+j] = ll1+2*ll2+4*ll4+8*ll8;
        
        int lr1 = select(0,1,(BF & 3)==3);
        int lr2 = select(0,1,(BF & 12)==4);
        int lr4 = select(0,1,(BF & 12)==8);
        int lr8 = select(0,1,(BF & 256)==256);
        tilingLR[i*N+j] = lr1+2*lr2+4*lr4+8*lr8;
        
        int lh1 = select(0,1,(BF & 3)==1);
        int lh2 = select(0,1,(BF & 3)==2);
        int lh4 = select(0,1,(BF & 12)==12);
        int lh8 = select(0,1,(BF & 256)==256);
        tilingLH[i*N+j] = lh1+2*lh2+4*lh4+8*lh8;
        
        
    }
    
}

__kernel void UpdateLozengeFromButterflysR(__global int* tilingBR,__global int* tilingLH,__global int* tilingLL,__global int* tilingLR, const int N)
{
    // update lozenges from right butterfly
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    
    if ( i < N-1 && j < N-1 && i > 0 && j > 0 ) {
        int BF = tilingBR[i*N+j];
        int lr1 = select(0,1,(BF & 3)==3);
        int lr2 = select(0,1,(BF & 12)==12);
        int lr4 = select(0,1,(BF & 48)==48);
        int lr8 = select(0,1,(BF & 192)==192);
        tilingLR[i*N+j] = lr1+2*lr2+4*lr4+8*lr8;
        
        int lh1 = select(0,1,(BF & 3)==1);
        int lh2 = select(0,1,(BF & 3)==2);
        int lh4 = select(0,1,(BF & 256)==256);
        int lh8 = select(0,1,(BF & 192)==192);
        tilingLH[i*N+j] = lh1+2*lh2+4*lh4+8*lh8;
        
        int ll1 = select(0,1,(BF & 3)==3);
        int ll2 = select(0,1,(BF & 256)==256);
        int ll4 = select(0,1,(BF & 192)==64);
        int ll8 = select(0,1,(BF & 192)==128);
        tilingLL[i*N+j] = ll1+2*ll2+4*ll4+8*ll8;
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


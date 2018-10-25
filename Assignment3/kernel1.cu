#include <stdio.h>
#include "kernel1.h"


//extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width) 
{
    extern __shared__ float s_data[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	i = i + 1;
	
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	j = j + 1;
	
    if(i < width - 1 || i > 1 || j < width - 1 || j > 1) {
		//copy memory to shared memory
		
		//perform the calculations
		
	}
}


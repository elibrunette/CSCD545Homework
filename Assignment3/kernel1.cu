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
    extern __shared__ float s_data[4][blockIdx.x];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	i = i + 1;
	
	int sharedMemIndex = threadIdx.x;

	int j = blockIdx.y;
	j = j + 1;
	
    if(i < width - 1 || i > 1 || j < width - 1 || j > 1) {
		//copy memory to shared memory
		if(sharedMemIndex == 1) {
			s_data[1][1] = 0.2f * g_dataA[ i    * pitch +  j   ]; //itself
			s_data[0][0] = 0.1f * g_dataA[(i-1) * pitch + (j-1)]; //NW
			s_data[0][1] = 0.1f * g_dataA[(i-1) * pitch +  j   ]; //N
			s_data[1][0] = 0.1f * g_dataA[ i    * pitch + (j+1)]; //W
			s_data[2][0] = 0.1f * g_dataA[(i+1) * pitch + (j-1)]; //SW
			s_data[2][1] = 0.1f * g_dataA[(i+1) * pitch +  j   ]; //S
		}
		else if(sharedMemIndex == width - 2) {
			s_data[1][width - 2] = 0.2f * g_dataA[ i    * pitch +  j   ]; //itself
			s_data[0][width - 2] = 0.1f * g_dataA[(i-1) * pitch +  j   ]; //N
			s_data[0][width - 1] = 0.1f * g_dataA[(i-1) * pitch + (j-1)]; //NE
			s_data[1][width - 1] = 0.1f * g_dataA[ i    * pitch + (j+1)]; //E
			s_data[2][width - 1] = 0.1f * g_dataA[(i+1) * pitch + (j+1)]; //SE
			s_data[2][width - 2] = 0.1f * g_dataA[(i+1) * pitch +  j   ]; //S
		}
		else {
			s_data[0][sharedMemIndex] = 0.1f * g_dataA[(i-1) * pitch +  j   ]; //N
			s_data[1][sharedMemIndex] = 0.2f * g_dataA[ i    * pitch +  j   ]; //itself
			s_data[2][sharedMemIndex] = 0.1f * g_dataA[(i+1) * pitch +  j   ]; //S
		}
		__syncthread();
		//perform the calculations

				

	}
}


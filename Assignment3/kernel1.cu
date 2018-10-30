#include <stdio.h>
#include "kernel1.h"


extern  __shared__  float sdata[];

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
	int sharedMaxIndex = blockDim.x + 1;
	int sharedMemIndex = threadIdx.x;

	int j = blockIdx.y;
	j = j + 1;
	
    if(i < width - 1 || i > 1 || j < width - 1 || j > 1) {
		//copy memory to shared memory
		if(sharedMemIndex == 1) {
			s_data[sharedMaxIndex 	  + 2] = 0.2f * g_dataA[ i    * floatpitch +  j   ]; //itself
			s_data[			    0] = 0.1f * g_dataA[(i-1) * floatpitch + (j-1)]; //NW
			s_data[			    1] = 0.1f * g_dataA[(i-1) * floatpitch +  j   ]; //N
			s_data[sharedMaxIndex 	  + 1] = 0.1f * g_dataA[ i    * floatpitch + (j+1)]; //W
			s_data[sharedMaxIndex * 2 + 1] = 0.1f * g_dataA[(i+1) * floatpitch + (j-1)]; //SW
			s_data[sharedMaxIndex * 2 + 2] = 0.1f * g_dataA[(i+1) * floatpitch +  j   ]; //S
		}
		else if(sharedMemIndex == width - 2) {
			s_data[2 * sharedMaxIndex - 2	 ] = 0.2f * g_dataA[ i    * floatpitch +  j   ]; //itself
			s_data[sharedMaxIndex - 2	 ] = 0.1f * g_dataA[(i-1) * floatpitch +  j   ]; //N
			s_data[sharedMaxIndex - 1	 ] = 0.1f * g_dataA[(i-1) * floatpitch + (j-1)]; //NE
			s_data[2 * sharedMaxIndex - 1	 ] = 0.1f * g_dataA[ i    * floatpitch + (j+1)]; //E
			s_data[3 * sharedMaxIndex - 1	 ] = 0.1f * g_dataA[(i+1) * floatpitch + (j+1)]; //SE
			s_data[3 * sharedMaxIndex - 2	 ] = 0.1f * g_dataA[(i+1) * floatpitch +  j   ]; //S
		}
		else {
			s_data[2 * sharedMaxIndex + sharedMemIndex] = 0.1f * g_dataA[(i-1) * floatpitch +  j   ]; //N
			s_data[2 * sharedMaxIndex + sharedMemIndex] = 0.2f * g_dataA[ i    * floatpitch +  j   ]; //itself
			s_data[3 * sharedMaxIndex + sharedMemIndex] = 0.1f * g_dataA[(i+1) * floatpitch +  j   ]; //S
		}
		__syncthreads();
		
		//perform the calculations
		s_data[4 * sharedMaxIndex + sharedMemIndex] =	s_data[				sharedMemIndex - 1] + //NW
								s_data[				sharedMemIndex    ] + //N
								s_data[				sharedMemIndex + 1] + //NE
								s_data[sharedMaxIndex + 	sharedMemIndex - 1] + //W
								s_data[sharedMaxIndex +		sharedMemIndex    ] + //itself
								s_data[sharedMaxIndex + 	sharedMemIndex + 1] + //E
								s_data[sharedMaxIndex * 2 + 	sharedMemIndex - 1] + //SW
								s_data[sharedMaxIndex * 2 + 	sharedMemIndex    ] + //S
								s_data[sharedMaxIndex * 2 + 	sharedMemIndex + 1];  //SE
		//copy to shared memory		
		int midRow = blockIdx.y + 1;
		int col = blockDim.x * blockIdx.x;

		g_dataB[midRow * width + col] = s_data[3];

	}
}


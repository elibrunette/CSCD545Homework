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
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	i = i + 1;
	j = j + 1;

	int sharedWidth = width + 2;
	int sharedMemIndex = threadIdx.x + 1;

//	int  boundary = floatpitch - width;
	if(i < width - 1 && j < width - 1 && i >= 1 && j >= 1)
	{
	
		//copy memory to shared memory
		
		if(j == 1) {
			s_data[sharedWidth 		+ sharedMemIndex		] = g_dataA[ i    * floatpitch +  j   ]; //itself
			s_data[				  sharedMemIndex	     - 1] = g_dataA[(i-1) * floatpitch + (j-1)]; //NW
			s_data[				  sharedMemIndex		] = g_dataA[(i-1) * floatpitch +  j   ]; //N
			s_data[sharedWidth  		+ sharedMemIndex 	     - 1] = g_dataA[ i    * floatpitch + (j-1)]; //W
			s_data[sharedWidth * 2 		+ sharedMemIndex 	     - 1] = g_dataA[(i+1) * floatpitch + (j-1)]; //SW
			s_data[sharedWidth * 2 		+ sharedMemIndex		] = g_dataA[(i+1) * floatpitch +  j   ]; //S
		}
		else if(j == width - 2 || threadIdx.x + 1 == blockDim.x) {
			s_data[sharedWidth 		+ sharedMemIndex	 	] = g_dataA[ i    * floatpitch +  j   ]; //itself
			s_data[				  sharedMemIndex		] = g_dataA[(i-1) * floatpitch +  j   ]; //N
			s_data[				  sharedMemIndex 	     + 1] = g_dataA[(i-1) * floatpitch + (j+1)]; //NE
			s_data[sharedWidth 		+ sharedMemIndex 	     + 1] = g_dataA[ i    * floatpitch + (j+1)]; //E
			s_data[sharedWidth * 2  	+ sharedMemIndex 	     + 1] = g_dataA[(i+1) * floatpitch + (j+1)]; //SE
			s_data[sharedWidth * 2  	+ sharedMemIndex		] = g_dataA[(i+1) * floatpitch +  j   ]; //S
		}
		else {
			s_data[			 sharedMemIndex	] = g_dataA[(i-1) * floatpitch +  j   ]; //N
			s_data[sharedWidth     + sharedMemIndex	] = g_dataA[ i    * floatpitch +  j   ]; //itself
			s_data[sharedWidth * 2 + sharedMemIndex	] = g_dataA[(i+1) * floatpitch +  j   ]; //S
		}

		__syncthreads();
		
		//perform the calculations
		g_dataB[i * floatpitch + j] = (	
						.1f * s_data[			  sharedMemIndex		- 1] + //NW
						.1f * s_data[			  sharedMemIndex		   ] + //N
						.1f * s_data[			  sharedMemIndex		+ 1] + //NE
						.1f * s_data[sharedWidth 	+ sharedMemIndex 		- 1] + //W
						.2f * s_data[sharedWidth 	+ sharedMemIndex		   ] + //itself
						.1f * s_data[sharedWidth 	+ sharedMemIndex 		+ 1] + //E
						.1f * s_data[sharedWidth * 2 	+ sharedMemIndex 		- 1] + //SW
						.1f * s_data[sharedWidth * 2 	+ sharedMemIndex		   ] + //S
						.1f * s_data[sharedWidth * 2 	+ sharedMemIndex 		+ 1] //SE
					      ) * .95f; 
	}

}


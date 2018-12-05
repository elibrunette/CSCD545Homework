#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ArrayUtils.h"

__global__ void gpuKNNSolution(int * d_classified, int * d_test, double * d_result, int classCol, int classRow, int testCol, int testRow);

double *  gpuKNN(int ** classified, int ** test, int classCol, int classRow, int testCol, int testRow) {
	//initialize host variables
	int classifiedN = classCol * classRow;
	int testN = testRow * testCol;
	int toReturnN = testRow * classRow;

	//1D storage
	int * h_classified = twoDToOneD(classified, classCol, classRow);
	int * h_test = twoDToOneD(test, testCol, testRow);

	//gpu memory
	int * d_test;
	int * d_classified;
	double * d_result;

	//memory allocation
	double * toReturn = (double *) calloc(toReturnN, sizeof(double));
	cudaMalloc(&d_classified, classifiedN * sizeof(int));
	cudaMalloc(&d_test, testN * sizeof(int));
	cudaMalloc(&d_result, toReturnN * sizeof(double));

	//copy memory
	cudaMemcpy(d_classified, h_classified, classifiedN * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_test, h_test, testN * sizeof(double), cudaMemcpyHostToDevice);

	//set up the block and grid dementions 
	dim3 grid, block;
	block.x = 16;
	block.y = 1;
	grid.x = ceil((float) testRow / block.x);
	grid.y = 1;

	//call kernel
	gpuKNNSolution<<<grid, block>>>(d_classified, d_test, d_result, classCol, classRow, testCol, testRow);

	//copy memory back
	cudaMemcpy(toReturn, d_result, sizeof(double) * toReturnN, cudaMemcpyDeviceToHost);

	//free memory that is only used in this method
	free(h_classified);
	free(h_test);
	cudaFree(d_classified);
	cudaFree(d_test);
	cudaFree(d_result);

	return toReturn;
}

__global__ void gpuKNNSolution(int * d_classified, int * d_test, double * d_result, int classCol, int classRow, int testCol, int testRow) {
	int x = 0;
	int y = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= testRow)
		return;

	for(x = 0; x < classRow; x++) {
		for(y = 0; y < classCol; y++) {
			d_result[i] = 1;
		}
	}
}









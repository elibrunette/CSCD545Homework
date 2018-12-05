#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ArrayUtils.h"
#include "mergeSortInterface.h"
#include "FileUtils.h"

__global__ void gpuKNNSolution(int * d_classified, int * d_test, double * d_result, int classCol, int classRow, int testCol, int testRow);

int **  gpuKNN(int ** classified, int ** test, int classCol, int classRow, int testCol, int testRow) {
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
	int ** indexes = createInt2DArray(classRow, testRow);

	//memory allocation
	double * toReturn = (double *) calloc(toReturnN, sizeof(double));
	cudaMalloc(&d_classified, classifiedN * sizeof(int));
	cudaMalloc(&d_test, testN * sizeof(int));
	cudaMalloc(&d_result, toReturnN * sizeof(double));

	//copy memory
	cudaMemcpy(d_classified, h_classified, classifiedN * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_test, h_test, testN * sizeof(int), cudaMemcpyHostToDevice);

	//set up the block and grid dementions 
	dim3 grid, block;
	block.x = 512;
	block.y = 1;
	grid.x = ceil(((float) testRow * classRow) / block.x);
	grid.y = 1;

	//printf("Made it to right before the kernel\n");
	//printf("ClassRow: %d\n TestRow: %d\n\n\n", classRow, testRow);
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

	//copy cpu from here using toReturn as the unsorted Distance array
	
	double ** sortedDistances = oneDToTwoD(toReturn, classRow, testRow);
	//printf("not sortedDistance array:\n");
	//print2DDoubleArray(sortedDistances, classRow, testRow);
	//printf("about to compute indexes in gpu\n");

	//indexes = mergeSortForKNN(sortedDistances, classCol, classRow, testCol, testRow);
	outputToFile("gpuDistance", sortedDistances, testRow, classRow);
	int x = 0; 
	int y = 0;
	for(x = 0; x < testRow; x++) {
		for( y = 0; y < classRow; y++) {
			indexes[x][y] = y;
		}
	}

	//printf("Sorted gpu indexes\n");
	//print2DArray(indexes, classRow, testRow);

	return indexes;
}

__global__ void gpuKNNSolution(int * d_classified, int * d_test, double * d_result, int classCol, int classRow, int testCol, int testRow) {
	int y = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int row = i / classRow;
	int col = i % classRow;
	double runningTotal = 0;
	double difference = 0;

	if(i >= testRow * classRow)
		return;

	//d_result[i] = (double) d_test[i];

	
	for(y = 0; y < testCol; y++) {
		difference = (double) d_classified[col * classCol + y] - (double) d_test[row * testCol + y];
		runningTotal = runningTotal + (difference) * (difference);
	}
	d_result[i] = sqrt(runningTotal);

}









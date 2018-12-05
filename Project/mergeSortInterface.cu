#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ArrayUtils.h"
#include "./MergeSort/merge_sort.cu"
#include "./MergeSort/mergeSort_common.h"

/*
double * mergeHelper(double * arr, int n);

double ** mergeSort(double ** arr, int classCol, int classRow, int testCol, int testRow) {
	int n = classRow;
	int x = 0; 
	for(x = 0; x < classRow; x++) {
		arr[x] = mergeHelper(arr[x], n);
	}
	return arr;
}

double * mergeHelper(double * arr, int n) {
	double * toReturn = (double *) calloc(n, sizeof(double));
	int i = 0; 
	int j = 0;
	int smallest;
	double temp;

	for(i = 0; i < n; i++) {
		smallest = i;
		for(j = 0; j < n; j++) {
			if(arr[j] < arr[smallest]) 
				smallest = j;
		}
		temp = arr[i];
		arr[i] = arr[smallest];
		arr[smallest] = temp;
	}

	free(arr);
	return toReturn;
}
*/

void fillValues(int * arr, int N) {
	uint x = 0; 
	for(x = 0; x < N; x++) {
		arr[x] = x;
	} 
}

int ** mergeSortForKNN(double ** arr, int classCol, int classRow, int testCol, int testRow) {

	int ** toReturn = createInt2DArray(classRow, testRow);

	//setting up variables for sorting 
	int N = pow(2, ceil(log(classRow)/log(2)));

	if(N < SHARED_SIZE_LIMIT)
		N = SHARED_SIZE_LIMIT;

	int *h_SrcVal = (int *) malloc(N * sizeof(int));

	fillValues(h_SrcVal, N);
	printSingleIntArray(h_SrcVal, N);

	int *d_SrcVal, *d_BufVal, *d_DstVal, *h_label_temp;
	double * d_SrcKey, * d_BufKey, * d_DstKey, * h_dist_temp;
	
	//allocating memory on GPU 
	cudaMalloc((void **)&d_SrcKey, N * sizeof(double));


	cudaMalloc((void **)&d_DstKey, N * sizeof(double));
	cudaMalloc((void **)&d_DstVal, N * sizeof(int));
	cudaMalloc((void **)&d_BufKey, N * sizeof(double));
	cudaMalloc((void **)&d_BufVal, N * sizeof(int));
	cudaMalloc((void **)&d_SrcVal, N * sizeof(int));
	
	initMergeSort();

	int i = 0;
	int j = 0;

	h_dist_temp = (double *) malloc(N * sizeof(double));
	h_label_temp = (int *) malloc(N * sizeof(int));

	for(i = 0; i < testRow; i++) { 		//iterates through the arr array and creates a temp array that gets padded
		for(j = 0; j < N; j++) {
			if(j < classRow) {
				h_dist_temp[j] = arr[i][j];
			}
			else {
				h_dist_temp[j] = INFINITY;
			}
		}

		cudaMemcpy(d_SrcKey, h_dist_temp, N * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(double), cudaMemcpyHostToDevice);
		
		printf("Made it to right before GPU MergeSort\n");
		printf("Value for N: %d\n", N);

		mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N, 1);

		cudaMemcpy(arr[i], d_DstKey, testRow * classRow * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(toReturn[i], d_DstVal, testRow * classRow * sizeof(int), cudaMemcpyDeviceToHost);
	}

	print2DDoubleArray(arr, classRow, testRow);
	print2DArray(toReturn, classRow, testRow);
	return toReturn;
}












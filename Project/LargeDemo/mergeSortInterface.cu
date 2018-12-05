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

	int *h_SrcVal = (int *) malloc(testRow * testCol * sizeof(int)); //h_label
	int ** toReturn = createInt2DArray(classRow, testRow);

    	double  *d_SrcKey, *d_BufKey, *d_DstKey, *h_dist_temp;
    	int     *d_SrcVal, *d_BufVal, *d_DstVal, *h_label_temp;
	//print2DDoubleArray(arr, classRow, testRow);
	//printf("Incoming perameters classCol: %d\nclassRow: %d\ntestCol: %d\ntestRow:%d\n", classCol, classRow, testCol, testRow);

	//setting up variables for sorting 
	int N = pow(2, ceil(log(classRow)/log(2)));

	if(N < SHARED_SIZE_LIMIT)
		N = SHARED_SIZE_LIMIT;

	fillValues(h_SrcVal, N);
	cudaMalloc((void **)&d_SrcKey, N * sizeof(double));
	cudaMalloc((void **)&d_DstKey, N * sizeof(double));
	cudaMalloc((void **)&d_BufKey, N * sizeof(double));
	cudaMalloc((void **)&d_SrcVal, N * sizeof(int));
	cudaMalloc((void **)&d_DstVal, N * sizeof(int));
	cudaMalloc((void **)&d_BufVal, N * sizeof(int));

	//printSingleIntArray(h_SrcVal, N);


	
	initMergeSort();

	int i = 0;
	int j = 0;

	h_dist_temp = (double *) malloc(N * sizeof(double));
	h_label_temp = (int *) malloc(N * sizeof(int));

	//printf("Made up to the double for loop for sorting the array\n");
	for(i = 0; i < testRow; i++) { 		//iterates through the arr array and creates a temp array that gets padded
			
		//allocating memory on GPU 

		//printf("Filled Values\n");



		//printf("i: %d\n", i);
		for(j = 0; j < N; j++) {
			if(j < classRow) {
				//printf("arr[i][j]: %f\n" , arr[i][j]);
				h_dist_temp[j] = arr[i][j];
				h_label_temp[j] = h_SrcVal[j];
			}
			else {
				h_dist_temp[j] = INFINITY; //10000 is out of range of the rng used for this program
				h_label_temp[j] = 0;
			}
		}
		//printSingleArray(h_dist_temp, N);

		cudaMemcpy(d_SrcKey, h_dist_temp, N * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_SrcVal, h_label_temp, N * sizeof(int), cudaMemcpyHostToDevice);
		
		//printf("Made it to right before GPU MergeSort\n");
		//printf("Value for N: %d\n", N);

        		mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N, 1);
		//printf("Right after the mergeSort\n");
		
		cudaMemcpy(arr[i], d_DstKey, classRow * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(toReturn[i], d_DstVal, classRow * sizeof(int), cudaMemcpyDeviceToHost);

		//printf("copying back to the result arr\n");
		//copy computed values back to original array
	/*		
		int x = 0;
		for(x = 0; x < classRow; x++) {
			arr[i][x] = h_dist_temp[x];
			toReturn[i][x] = h_SrcVal[x];
		}
	*/
	}

	//printf("made it to right after the for loop\n");
	closeMergeSort();

    	cudaFree(d_SrcKey);
    	cudaFree(d_DstKey);
    	cudaFree(d_BufKey);
    	cudaFree(d_DstVal);
    	cudaFree(d_SrcVal);
    	cudaFree(d_BufVal);

    	free(h_dist_temp);
    	free(h_label_temp);

	//printf("Made it out of merge and right before the print arr command");
	//print2DDoubleArray(arr, classRow, testRow);
	//print2DArray(toReturn, classRow, testRow);
	return toReturn;
}












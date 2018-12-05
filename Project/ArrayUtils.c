#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void printSingleIntArray(int * arr, int n) {
	int x = 0;
	printf("[");
	for(x = 0; x < n; x++) {
		(x ==(n - 1)) ? printf("%d", arr[x]) : printf("%d ", arr[x]);
	}
	printf("]\n");
}

void printSingleArray(double * arr, int n) {
	int x = 0;
	printf("["); 
	for( x = 0; x < n; x++) {
		(x == (n - 1)) ? printf("%f", arr[x]) : printf("%f ", arr[x]);
	}
	printf("]/n");
}

void print2DArray(int ** arr, int col, int row) {
	int x = 0; 
	int y = 0; 
	printf("[\n");
	for(x = 0; x < row; x++) {
		printf("[");
		for(y = 0; y < col; y++) {
			(y == (col - 1)) ? printf("%d", arr[x][y]) : printf("%d ", arr[x][y]);
		}
		printf("]\n");
	}
	printf("]\n");
}

void print2DDoubleArray(double ** arr, int col, int row) {
	int x = 0; 
	int y = 0; 
	printf("[\n");
	for(x = 0; x < row; x++) {
		printf("[");
		for(y = 0; y < col; y++) {
			(y == (col - 1)) ? printf("%f", arr[x][y]) : printf("%f ", arr[x][y]);
		}
		printf("]\n");
	}
	printf("]\n");
}

void freeIntDoublePointer(int ** input, int rows) {
	int i = 0;
	for(i = 0; i < rows; i++) {
		free(input[i]);
		//printf("Freed something\n");
	}
	//printf("Made it through for loop\n\n\n\n\n");
	free(input);
}

int * twoDToOneD(int ** arr, int col, int row) {
	int * toReturn = (int *) calloc(col * row, sizeof(int));
	int x = 0;
	int y = 0;
	int i = 0;

	for(x = 0; x < row; x++) {
		for(y = 0; y < col; y++) {
			toReturn[i++] = arr[x][y]; 
		}
	}

	return toReturn;
}

double ** createDouble2DArray(int col, int row) {
	int x = 0;

	double ** toReturn = (double **) calloc(row + 1, sizeof(double *));
	
	for(x = 0; x < row; x++) {
		toReturn[x] = (double *) calloc(col + 1, sizeof(double));
	}
	return toReturn;
}

int ** createInt2DArray(int col, int row) {
	int x = 0; 

	int ** toReturn = (int **) calloc(row + 1, sizeof(double));

	for(x = 0; x < row; x++) {
		toReturn[x] = (int *) calloc(col + 1, sizeof(int));
	}
	return toReturn;
}

double ** oneDToTwoD(double * arr, int col, int row) {
	int x = 0;
	int y = 0;
	int i = 0; 
	int n = col * row;
	//intialize 2D arr
	double ** toReturn = createDouble2DArray(col, row);

	for(x = 0; x < row; x++) {
		for(y = 0; y < col; y++) {
			toReturn[x][y] = arr[i++];
		}
	}
	free(arr); 
	return toReturn;
}


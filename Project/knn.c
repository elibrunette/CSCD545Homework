#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//#include "gpuKNN.h"
#include "FileUtils.h"

void usage() {
	printf("Usage: ./knn numberOfNeighbors inputClassificationsFile inputTestFile outputClassificationsFile\n");
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

void freeIntDoublePointer(int ** input, int rows) {
	int i = 0;
	for(i = 0; i < rows; i++) {
		free(input[i]);
		printf("Freed something\n");
	}
	printf("Made it through for loop\n\n\n\n\n");
	free(input);
}

int main(int argc, char * argv[]) {
	//step zero: check the input from the user and store input from the user
	if(argc != 5)
		usage();

	//initalize variables
	char * initialSetup = argv[2];
	char * testCases = argv[3];
	char * outputFile = argv[4];
	int neighbors = atoi(argv[1]);
	//printf("initialSetup: %s\n\n\n", argv[2]);

	int ** classifiedPoints;
	FILE * initialFin = fopen(initialSetup, "r");
	int * classifiedPointsHeader = getHeader(initialFin);

	int ** testPoints;
	FILE * test = fopen(testCases, "r");
	int * testPointsHeader = getHeader(test);

	//Step one: read in file for orginal values
	classifiedPoints = readFile(initialFin, classifiedPointsHeader[0], classifiedPointsHeader[1]);
	testPoints = readFile(test, testPointsHeader[0], testPointsHeader[1]);

	//step two: create the kernel

	//step three: sort the array

	//step four: count the lowest n neighbors

	//step five: free up memory
	freeIntDoublePointer(classifiedPoints, classifiedPointsHeader[1]);
	freeIntDoublePointer(testPoints, testPointsHeader[1]);
	fclose(test);
	fclose(initialFin);
	
	return 0;
}

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "gpuKNN.h"
#include "cpuKNN.h"
#include "FileUtils.h"
#include "ArrayUtils.h"
#include "./MergeSort/mergeSort_common.h"

void usage() {
	printf("Usage: ./knn numberOfNeighbors inputClassificationsFile inputTestFile outputClassificationsFile\n");
	exit(-1);
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
	const char * cpuOutput = "cpuIntermediateResult";
	//printf("initialSetup: %s\n\n\n", argv[2]);

	//variables for step one
	int ** classifiedPoints;
	FILE * initialFin = fopen(initialSetup, "r");
	int * classifiedPointsHeader = getHeader(initialFin);
	int classCol = classifiedPointsHeader[0];
	int classRow = classifiedPointsHeader[1];

	int ** testPoints;
	FILE * test = fopen(testCases, "r");
	int * testPointsHeader = getHeader(test);
	int testCol = testPointsHeader[0];
	int testRow = testPointsHeader[1];
	
	//variables for step two
	int ** cpuIndexes;
	double * gpuDistanceArray;
	int * cpuClassified;

	//Step one: read in file for orginal values
	classifiedPoints = readFile(initialFin, classCol, classRow);
	testPoints = readFile(test, testCol, testRow);

	//step 2a: create a cpuSolution
	cpuIndexes = cpuKNNSolution(classifiedPoints, testPoints, classCol, classRow, testCol, testRow);
	//print2DDoubleArray(cpuDistanceArray, classifiedPointsHeader[1], testPointsHeader[1]);
	//outputToFile(cpuOutput, cpuDistanceArray, testPointsHeader[1], classifiedPointsHeader[1]);
	cpuClassified = getClassifications(cpuIndexes, classifiedPoints, classRow, classCol, testRow, neighbors);
	

	//step 2b: create a gpuSolution
	//gpuDistanceArray = gpuKNN(classifiedPoints, testPoints, classifiedPointsHeader[0], classifiedPointsHeader[1], testPointsHeader[0], testPointsHeader[1]);
	//double ** distance = oneDToTwoD(gpuDistanceArray, classifiedPointsHeader[1], testPointsHeader[1]);
	//print2DDoubleArray(distance, classifiedPointsHeader[1], testPointsHeader[1]);

	//step three: sort the array

	//step four: count the lowest n neighbors

	//step five: output to file
	

	//step six: free up memory
	freeIntDoublePointer(classifiedPoints, classifiedPointsHeader[1]);
	freeIntDoublePointer(testPoints, testPointsHeader[1]);
	fclose(test);
	fclose(initialFin);
	
	return 0;
}

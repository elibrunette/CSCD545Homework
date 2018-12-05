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

	int ** testPoints;
	FILE * test = fopen(testCases, "r");
	int * testPointsHeader = getHeader(test);

	//variables for step two
	double ** cpuDistanceArray;
	double * gpuDistanceArray;

	//Step one: read in file for orginal values
	classifiedPoints = readFile(initialFin, classifiedPointsHeader[0], classifiedPointsHeader[1]);
	testPoints = readFile(test, testPointsHeader[0], testPointsHeader[1]);

	//step 2a: create a cpuSolution
	cpuDistanceArray = cpuKNNSolution(classifiedPoints, testPoints, classifiedPointsHeader[0], classifiedPointsHeader[1], testPointsHeader[0], testPointsHeader[1]);
	//print2DDoubleArray(cpuDistanceArray, classifiedPointsHeader[1], testPointsHeader[1]);
	//outputToFile(cpuOutput, cpuDistanceArray, testPointsHeader[1], classifiedPointsHeader[1]);

	

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

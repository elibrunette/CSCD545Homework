#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "gpuKNN.h"
#include "cpuKNN.h"
#include "FileUtils.h"
#include "ArrayUtils.h"
#include "./Timing/timing.h"
#include "./MergeSort/mergeSort_common.h"

void usage() {
	printf("Usage: ./knn numberOfNeighbors inputClassificationsFile inputTestFile  cpuOutputClassificationsFile gpuOutputClassificationsFile\n");
	exit(-1);
}

int main(int argc, char * argv[]) {
	//step zero: check the input from the user and store input from the user
	if(argc != 6)
		usage();

	//initalize variables
	char * initialSetup = argv[2];
	char * testCases = argv[3];
	char * cpuOutput = argv[4];
	char * gpuOutput = argv[5];
	int neighbors = atoi(argv[1]);
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
	int ** gpuIndexes;
	int * cpuClassified;
	int * gpuClassified;

	//variables for calculating the time
	double now, then;
	double cpuTime, gpuTime;

	//Step one: read in file for orginal values
	classifiedPoints = readFile(initialFin, classCol, classRow);
	testPoints = readFile(test, testCol, testRow);

	//step 2a: create a cpuSolution
	//warm up kernel
	cpuKNNSolution(classifiedPoints, testPoints, classCol, classRow, testCol, testRow);

	then = currentTime();
	cpuIndexes = cpuKNNSolution(classifiedPoints, testPoints, classCol, classRow, testCol, testRow);
	cpuClassified = getClassifications(cpuIndexes, classifiedPoints, classRow, classCol, testRow, neighbors);
	now = currentTime();
	cpuTime = now - then;
	outputFinal(cpuOutput, classifiedPoints, cpuClassified, testCol, testRow, cpuTime);

	//step 2b: create a gpuSolution
	then = currentTime();
	gpuIndexes = gpuKNN(classifiedPoints, testPoints, classifiedPointsHeader[0], classifiedPointsHeader[1], testPointsHeader[0], testPointsHeader[1]);
	gpuClassified = getClassifications(gpuIndexes, classifiedPoints, classRow, classCol, testRow, neighbors);
	now = currentTime();
	gpuTime = now - then;
	printf("gpuTime %f\ncpuTime %f\n", gpuTime, cpuTime);
	printf("speedup (cpuTime/gpuTime): %f\n", cpuTime/gpuTime);
	outputFinal(gpuOutput, classifiedPoints, gpuClassified, testCol, testRow, gpuTime);

	//step six: free up memory
	freeIntDoublePointer(classifiedPoints, classifiedPointsHeader[1]);
	freeIntDoublePointer(testPoints, testPointsHeader[1]);
	fclose(test);
	fclose(initialFin);
	
	return 0;
}

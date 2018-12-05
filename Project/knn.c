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
	printf("Usage: ./knn numberOfNeighbors numberOfTestRow numberOfTestColumns numberOfClassifiedColumns cpuOutput gpuOutput\n");
	exit(-1);
}

int main(int argc, char * argv[]) {
	//step zero: check the input from the user and store input from the user
	if(argc <6 || argc >7)
		usage();

	int flag = 0;
	//initalize variables
	int neighbors = atoi(argv[1]);
	char * initialSetup;
	char * testCases;
	char * cpuOutput;
	char * gpuOutput;
	int * classifiedPointsHeader;
	int * testPointsHeader;
	FILE * initialFin;
	FILE * test;

	int ** classifiedPoints;
	int ** testPoints;

	int testCol, testRow, classCol, classRow;

	if(argc == 6) {
		initialSetup = argv[2];
		testCases = argv[3];
		cpuOutput = argv[4];
		gpuOutput = argv[5];

		//variables for step one

		initialFin = fopen(initialSetup, "r");
		classifiedPointsHeader = getHeader(initialFin);
		classCol = classifiedPointsHeader[0];
		classRow = classifiedPointsHeader[1];


		test = fopen(testCases, "r");
		testPointsHeader = getHeader(test);
		testCol = testPointsHeader[0];
		testRow = testPointsHeader[1];

		//Step one: read in file for orginal values
		classifiedPoints = readFile(initialFin, classCol, classRow);
		testPoints = readFile(test, testCol, testRow);
		flag = 1;

	}else if(argc == 7) {
		testRow = atoi(argv[2]);
		testCol = atoi(argv[3]);
		classRow = atoi(argv[4]);
		classCol = testCol + 1;
		cpuOutput = argv[5];
		gpuOutput = argv[6];

		//testPoints = createTestPoints(testRow, testCol);
		//classifiedPoints = createClassifiedPoints(classRow, classCol);
	}else {
		usage();
	}
	
	//printf("initialSetup: %s\n\n\n", argv[2]);

	
	//variables for step two
	int ** cpuIndexes;
	double * gpuDistanceArray;
	int ** gpuIndexes;
	int * cpuClassified;
	int * gpuClassified;

	//variables for calculating the time
	double now, then;
	double cpuTime, gpuTime;


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
	gpuIndexes = gpuKNN(classifiedPoints, testPoints, classCol, classRow, testCol, testRow);
	gpuClassified = getClassifications(gpuIndexes, classifiedPoints, classRow, classCol, testRow, neighbors);
	now = currentTime();
	gpuTime = now - then;
	printf("gpuTime %f\ncpuTime %f\n", gpuTime, cpuTime);
	printf("speedup (cpuTime/gpuTime): %f\n", cpuTime/gpuTime);
	outputFinal(gpuOutput, classifiedPoints, gpuClassified, testCol, testRow, gpuTime);

	//step six: free up memory
	freeIntDoublePointer(classifiedPoints, classRow);
	freeIntDoublePointer(testPoints, testRow);

	if(flag == 1) {
		fclose(test);
	}

	fclose(initialFin);
	
	return 0;
}

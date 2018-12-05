#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "ArrayUtils.h"
#include "mergeSortInterface.h"

double distance(int * classif, int * test, int n);

int * getClassifications(int ** indexes, int ** classifiedPoints, int classRow, int testRow);

int ** cpuKNNSolution(int ** classifiedPoints, int ** testPoints, int classCol, int classRow, int testCol, int testRow) {
	
	double * temp = (double *) calloc( classRow * testRow, sizeof(double));
	int ** indexes;
	int i = 0;
	int x = 0; 
	int y = 0; 
	double dist = 0;

	for(x = 0; x < testRow; x++) {
		for(y = 0; y < classRow; y++) {
			temp[i++] = distance(classifiedPoints[y], testPoints[x], testCol);
		}
		printf(" "); //I HAVE NO IDEA WHY YOU HAVE TO BE HERE
	}
	double ** sortedDistances = oneDToTwoD(temp, classRow, testRow);
	//printf("notSortedDistance array:\n");
	//print2DDoubleArray(sortedDistances, classRow, testRow);
	//printf("right before mergeSort in cpuKNNSolution\n");

	indexes = mergeSortForKNN(sortedDistances, classCol, classRow, testCol, testRow);

	//printf("\n\n\nsortedDistances array: /n");
	//print2DDoubleArray(sortedDistances, classRow, testRow);
	//printf("Indexes for sorted array:\n");
	//print2DArray(indexes, classRow, testRow);

	//find the classified values to return

	return indexes;
}

double distance(int * classif, int * test, int n) {
	int x = 0; 
	int i = 0; 
	int runningTotal = 0;
	int temp = 0;

	for(x = 0; x < n; x++) {
		temp = classif[x] - test[x];
		runningTotal =runningTotal + (temp * temp);
	}
	return sqrt((double) runningTotal);
}

int * getClassifications(int ** indexes, int ** classifiedPoints, int classRow, int classCol, int testRow, int k) {
	int * toReturn = (int *) calloc(testRow, sizeof(int));
	int * count = (int *) calloc(classRow, sizeof(int));
	int x = 0; 
	int y = 0;
	int z = 0;
	int maxValue = -1;
	int maxIndex = -1;
	int indexForCount = -1;
	for(x = 0; x < testRow; x++) {
		for(y = 0; y < k; y++) {
			count[classifiedPoints[indexes[x][y]][classCol - 1]]++ ;
		}
		maxValue = 0;
		maxIndex = 0;
		for(z = 0; z < classRow; z++) {
			if(maxValue < count[z]) {
				maxValue = count[z];
				maxIndex = z;
			}
		}
		printSingleIntArray(count, classRow);
		toReturn[x] = maxIndex;
		//reset count
		for(y = 0; y < classRow; y++) {
			count[y] = 0;
		}
	}

	printSingleIntArray(toReturn, testRow);
	return toReturn;
}

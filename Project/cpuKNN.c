#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "ArrayUtils.h"
#include "mergeSortInterface.h"

double distance(int * classif, int * test, int n);


double ** cpuKNNSolution(int ** classifiedPoints, int ** testPoints, int classCol, int classRow, int testCol, int testRow) {
	
	double * temp = (double *) calloc( classRow * testRow, sizeof(double));
	int ** indexes;
	int i = 0;
	int x = 0; 
	int y = 0; 
	double dist = 0;

	for(x = 0; x < testRow; x++) {
		for(y = 0; y < classRow; y++) {
			dist = distance(classifiedPoints[y], testPoints[x], testCol);
			printf("%f  ", dist);
			temp[i++] = dist;
		}
		printf("\n");
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

	return sortedDistances;
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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "ArrayUtils.h"

double distance(int * classif, int * test, int n);


double ** cpuKNNSolution(int ** classifiedPoints, int ** testPoints, int classCol, int classRow, int testCol, int testRow) {
	double ** toReturn = createDouble2DArray(testRow, testRow);
	double * temp = (double *) calloc(classRow * testRow, sizeof(double));
	int i = 0;
	int x = 0; 
	int y = 0; 

	for(x = 0; x < testRow; x++) {
		for(y = 0; y < classRow; y++) {
			temp[i++] = distance(classifiedPoints[y], testPoints[x], testCol);
		}
	}

	toReturn = oneDToTwoD(temp, classRow, testRow);

	return toReturn;
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
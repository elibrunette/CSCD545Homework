
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#include "pgmUtility.h"
#include "pgmProcess.h"

// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: You can NOT change the input, output, and argument type of the functions in pgmUtility.h
// NOTE: You can NOT change the prototype of any functions listed in pgmUtility.h

int ** pgmRead( char **header, int *numRows, int *numCols, FILE *in  ) {
	
	int counter = 0;
	char str[500];
	int rowCount = 0;
	int colCount = 0;
	char * token;
	char * remainder;
	int toAdd;
	int ** toReturn = (int ** ) calloc((* numRows), sizeof(int *));

	int totalCounter = 0;
	//initialize int **
	for(counter = 0; counter < (* numRows ); counter++) {
		toReturn[counter] = (int *) calloc((* numCols) + 1, sizeof(int));
	}
	while( fgets(str, 500, in) != NULL) {
		remainder = str;
		while(token = strtok_r(remainder, " ", &remainder)) {
			if(strcmp(token, "\n") != 0) {
				toAdd = atoi(token);
				toReturn[rowCount][colCount] = toAdd;
				colCount++;
				totalCounter++;
				if(colCount == (*numCols)) {
					colCount = 0;
					rowCount++;
				}
			}
		}
	}
	
	
	return toReturn;
}

int pgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow,
                  int centerCol, int radius, char **header ) {
	int x = 0; 
	int y = 0; 

	//cpu solution
	for( x = 0; x < numRows; x++) {
		for(y = 0; y < numCols; y++) {
			if(calcDistance(x, y, centerRow, centerCol) < radius)
				pixels[x][y] = 0;
		}
	}

	//gpu solution
	gpuCircleSolution(pixels, numRows, numCols, centerRow, centerCol, radius);

	return 0;
}

/**
*  Return distance between points
**/

int calcDistance( int x1, int y1, int x2, int y2 ) {
	double diffX = (x1 - x2) * (x1 - x2);
	double diffY = (y1 - y2) * (y1 - y2);
	return (sqrt(diffX + diffY));
}

int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header ) {
	//CPU Solution
	/*
	int x = 0;
	int y = 0; 
	for(x = 0; x < numRows; x++) {
		for(y = 0; y < numCols; y++) {
			if(x < edgeWidth || x >= (numRows - edgeWidth) || y < edgeWidth || y >= (numCols - edgeWidth))
				pixels[x][y] = 0;
		}
	}
	*/
	
	//GPU Solution 
	//gpuSolutionEdge(pixels, numRows, numCols, edgeWidth);

	gpuEdgeSolution(pixels, numRows, numCols, edgeWidth);
	return 0;
}

int pgmDrawLine( int **pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ) {
    	int row = 0; 
	int col = 0; 
	double slope;
	double intercept;

/*
	if(p1col != p2col) {
		slope = getSlope(p1col, p1row, p2col, p2row);
		intercept = getIntercept(slope, p1col, p1row);
		for(row = 0; row < numRows; row++) {
			for(col = 0; col < numCols; col++) {
				if(ifInLine(slope, intercept, col, row) == 0) //checks to see if the 'y' matches 
					if(col <= max(p1col, p2col) && col >= min(p1col, p2col)) //checks to see if the 'x' matches
						pixels[row][col] = 0;
			}
		}
	} 
	else if (p1col == p2col) {
		for(row = 0; row < numRows; row++) {
			if(row >= min(p1row, p2row) && row <= max(p1row, p2row))
				pixels[row][p1col] = 0;
		}
	}
*/

	gpuLineSolution(pixels, numRows, numCols, p1row, p1col, p2row, p2col);

	return 0;
}

//Returns the value of the slope
double getSlope(int x1, int y1, int x2, int y2) {
	double diffNumerator = (y2 - y1);
	double diffDenominator = (x2 - x1);

	return diffNumerator / diffDenominator;
}
//Returns the value of the intercept
double getIntercept(double slope, int x1, int y1) {
	return y1 - (slope * x1);
}

//returns 0 if point in the line, otherwise returns 1
int ifInLine(double slope, double intercept, int x, int y) {
	double testIntercept = y - (x * slope);
	double epsilon = .5;
	double result = testIntercept - intercept;
	
	if(abs(result) < epsilon)
		return 0;

	return 1;
}

int min(int x1, int x2) {
	if( x2 < x1)
		return x2;
	return x1;
}

int max(int x1, int x2) {
	if( x2 > x1 ) 
		return x2;
	return x1;
}

int pgmWrite( const char **header, const int **pixels, int numRows, int numCols, FILE *out ) {
	int i = 0;
	int x = 0;
	int y = 0;
	char * temp;
	int counter = 1;
	
	for(i = 0; i < 4; i++) {
		fprintf(out, "%s", header[i]);
	}

	for(x = 0; x < numRows; x++) {
		for(y = 0; y < numCols; y++) {
			if(counter < 14) {
				fprintf(out, "%d ", pixels[x][y]);			}
			else {
				fprintf(out, "%d\n", pixels[x][y]);
				counter = 0;
			}
			counter++;
		}
		fprintf(out, "\n");
		counter = 1;
	}
	if(fclose(out) == 0) {
		return 0;
	}	

	return -1;
}

int * convertArrayToSingle(int ** pixels, int numRows, int numCols) {
	int row = 0; 
	int col = 0; 
	int i = 0;
	int * toReturn = (int *) calloc(numRows * numCols, sizeof(int));
	
	for(row = 0; row < numRows; row++) {
		for(col = 0; col < numCols; col++) {
			toReturn[i++] = pixels[row][col];
		}
	}
	return toReturn;
}

void copyArrayBack(int ** pixels, int * toReturn, int numRows, int numCols) {
	int row = 0; 
	int col = 0; 
	int i = 0; 
	
	
	for(row = 0; row < numRows; row++) {
		for(col = 0; col < numCols; col++) {
			pixels[row][col] = toReturn[i++]; 
		}
	}
}


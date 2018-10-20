
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "pgmUtility.h"

// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: You can NOT change the input, output, and argument type of the functions in pgmUtility.h
// NOTE: You can NOT change the prototype of any functions listed in pgmUtility.h


/**
 *  Function Name: 
 *      pgmRead()
 *      pgmRead() reads in a pgm image using file I/O, you have to follow the file format. All code in this function are exectured on CPU.
 *      
 *  @param[in,out]  header  holds the header of the pgm file in a 2D character array
 *                          After we process the pixels in the input image, we write the origianl 
 *                          header (or potentially modified) back to a new image file.
 *  @param[in,out]  numRows describes how many rows of pixels in the image.
 *  @param[in,out]  numCols describe how many pixels in one row in the image.
 *  @param[in]      in      FILE pointer, points to an opened image file that we like to read in.
 *  @return         If successful, return all pixels in the pgm image, which is an int **, equivalent to
 *                  a 2D array. Otherwise null.
 *
 */
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


/**
 *  Function Name:
 *      pgmDrawCircle()
 *      pgmDrawCircle() draw a circle on the image by setting relavant pixels to Zero.
 *                      In this function, you have to invoke a CUDA kernel to perform all image processing on GPU.
 *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 2D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      centerCol specifies at which column you like to center your circle.
 *  @param[in]      centerRow specifies at which row you like to center your circle.
 *                        centerCol and centerRow defines the center of the circle.
 *  @param[in]      radius    specifies what the radius of the circle would be, in number of pixels.
 *  @param[in,out]  header returns the new header after draw. 
 *                  the circle draw might change the maximum intensity value in the image, so we
 *                  have to change maximum intensity value in the header accordingly.
 *  @return         return 1 if max intensity is changed, otherwise return 0;
 */
int pgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow,
                  int centerCol, int radius, char **header ) {
	int x = 0; 
	int y = 0; 

	for( x = 0; x < numRows; x++) {
		for(y = 0; y < numCols; y++) {
			if(calcDistance(x, y, centerRow, centerCol) < radius)
				pixels[x][y] = 0;
		}
	}


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

/**
 *  Function Name:
 *      pgmDrawEdge()
 *      pgmDrawEdge() draws a black edge frame around the image by setting relavant pixels to Zero.
 *                    In this function, you have to invoke a CUDA kernel to perform all image processing on GPU.
 *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 2D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      edgeWidth specifies how wide the edge frame would be, in number of pixels.
 *  @param[in,out]  header returns the new header after draw.
 *                  the function might change the maximum intensity value in the image, so we
 *                  have to change the maximum intensity value in the header accordingly.
 *
 *  @return         return 1 if max intensity is changed by the drawing, otherwise return 0;
 */
int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header ) {
	//CPU Solution
	int x = 0;
	int y = 0; 
	for(x = 0; x < numRows; x++) {
		for(y = 0; y < numCols; y++) {
			if(x < edgeWidth || x >= (numRows - edgeWidth) || y < edgeWidth || y >= (numCols - edgeWidth))
				pixels[x][y] = 0;
		}
	}
	
	//GPU Solution 
	int num_bytes = maxRows * maxCol * sizeof(int);
	int * h_pixels = convertArrayToSingle(pixels, numRows, numCols); 
	
	dim3 grid, block;
	block_x = 32;
	block_y = 32;
	grid.x = ciel((float) maxCol / block.x);
	grid.y = ciel((float) maxRow / block.y);
	
	int * d_pixels = 0;
	cudaMalloc((void **) &d_pixels, num_bytes);
	cudaMemcpy(d_pixels, h_pixels, num_bytes, cudaMemcpyHostToDevice);
	
	//Kernel call
	GPUEdge<<<grid, block>>>(d_pixels, maxRow, maxCol,edgeWidth);
	
	cudaMemcpy(h_pizels, d_pixels, num_bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);
	
	copyArrayBack(pixels, h_pixels, numRows, numCols);//TODO!!!!!!! 
	free(h_pixels);
	return 0;
}



/**
 *  Function Name:
 *      pgmDrawLine()
 *      pgmDrawLine() draws a straight line in the image by setting relavant pixels to Zero.
 *                      In this function, you have to invoke a CUDA kernel to perform all image processing on GPU.
 *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 2D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      p1row specifies the row number of the start point of the line segment.
 *  @param[in]      p1col specifies the column number of the start point of the line segment.
 *  @param[in]      p2row specifies the row number of the end point of the line segment.
 *  @param[in]      p2col specifies the column number of the end point of the line segment.
 *  @param[in,out]  header returns the new header after draw.
 *                  the function might change the maximum intensity value in the image, so we
 *                  have to change the maximum intensity value in the header accordingly.
 *
 *  @return         return 1 if max intensity is changed by the drawing, otherwise return 0;
 */
int pgmDrawLine( int **pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ) {
    	int row = 0; 
	int col = 0; 
	double slope;
	double intercept;
	//printf("made it to draw lines method\n\n");
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

/**
 *  Function Name:
 *      pgmWrite()
 *      pgmWrite() writes headers and pixels into a pgm image using file I/O.
 *                 writing back image has to strictly follow the image format. All code in this function are exectured on CPU.
 *
 *  @param[in]  header  holds the header of the pgm file in a 2D character array
 *                          we write the header back to a new image file on disk.
 *  @param[in]  pixels  holds all pixels in the pgm image, which a 2D integer array.
 *  @param[in]  numRows describes how many rows of pixels in the image.
 *  @param[in]  numCols describe how many columns of pixels in one row in the image.
 *  @param[in]  out     FILE pointer, points to an opened text file that we like to write into.
 *  @return     return 0 if the function successfully writes the header and pixels into file.
 *                          else return -1;
 */
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
	
	for(row = 0; row < maxRow; row++) {
		for(col = 0; col < maxCol; col++) {
			toReturn[i++] = pixels[row][col];
		}
	}
}

void copyArrayBack(int ** pixels, int * toReturn, int numRows, int numCols) {
	int row = 0; 
	int col = 0; 
	int i = 0; 
	
	
	for(row = 0; row < maxRow; row++) {
		for(col = 0; col < maxCol; col++) {
			pixels[row][col] = toReturn[i++]; 
		}
	}
}


#include <math.h>

/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] ) {
    double xDistance = p2[0] - p1[0];
    double yDistance = p2[1] - p1[1];
    double square = xDistance * xDistance + yDistance * yDistance;
    return sqrt(square);
}

//GPU solution for the conpute edge case 
__device__ void GPUEdge(int * d_pixels, maxRow, maxCol,edgeWidth) {

	rowNum = blockIdx.y * blockDim.y + threadIdx.y;
	colNum = blockIdx.x * blockDix.x + threadIdx.x;
	int i = rowNum + colNum
	
	if(colNum < maxCol && rowNum < maxRow) {
		if(colNum >= (maxCol - edgeWidth) || colNum < edgeWidth || rowNum >= (maxRow - edgeWidth) || rowNum < edgeWidth) {
			d_pixels[i] = 0;
		}
	}
}

//GPU solution for the computed circle case 
__device__ void GPUCircle(int * d_pixels, int cRow, int cCol, int radius, int maxRow, int maxCol) {
	row = blockIdx.y * blockDim.y + threadIdx.y;
	col = blockIdx.x * blockDim.x + threadIdx.x;
	i = row + col;
	int p1[2] = {cRow, cCol};
	int p2[2] = {row, col);
	
	if(row < maxRow && col < maxCol) {
		if(calcDistance(p1, p2) < radius) {
			d_pixels[i] = 0;
		}
	}
}

//GPU solution for the line
__device__ void GPULine(int * d_pixels, int maxRow, int maxCol, int p1row, int p1col, int p2row, int p2col) {
	row = blockIdx.y * blockDim.y + threadIdx.y;
	col = blockIdx.x * blockDim.x + threadIdx.x;
	i = row + col;
	
	double slope;
	double intercept;
	
	if(row < maxRow && col < maxCol) {
		if(p1col != p2col) {
			slope = calcSlope(p1row, p1col, p2row, p2col);
			intercept = calcIntercept(p1col, p1row, slope);
			double b = calcIntercept(col, row, slope);
			if(abs(b - intercept) < .5) {
				d_pixels[i] = 0;
			}
		}
		else {
			if(col == p1col && row < max(p1row, p2row) && row > min(p1row, p2row)) {
				d_pixels[i] = 0;
			}
		}
	}
}

__device__ double calcSlope(int p1row, int p1col, int p2row, int p2col) {
	double denominator = (p2col - p1col);
	double numerator = (p2row - p1col);
	
	return (denominator / numerator);
}


__device__ double calcIntercept( int col, int row, slope) {
	return col - (row * slope);
}

__device__ int min(int x, int y) {
	if(x < y) 
		return x;
	
	return y;
}

__device__ int max(int x, int y) {
	if(x > y) 
		return x;
		
	return y;
}
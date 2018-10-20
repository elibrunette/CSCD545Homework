#include <math.h>
#include <stdio.h>

#include "pgmProcess.h"
#include "pgmUtility.h"

__host__ void gpuEdgeSolution(int ** pixels, int numRows, int numCols, int edgeWidth ){
	size_t num_bytes = numRows * numCols * sizeof(int);
	int * h_pixels = convertArrayToSingle(pixels, numRows, numCols); 
	int * d_pixels;
	cudaMalloc(&d_pixels, num_bytes);
	cudaMemcpy(d_pixels, h_pixels, num_bytes, cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = ceil((float) numCols / block.x);
	grid.y = ceil((float) numRows / block.y);


	//Kernel call
	GPUEdge<<<grid, block>>>(d_pixels, numRows, numCols, edgeWidth);
	cudaMemcpy(h_pixels, d_pixels, num_bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);
	
	copyArrayBack(pixels, h_pixels, numRows, numCols); 
	free(h_pixels);

}

//GPU solution for the conpute edge case 
__global__ void GPUEdge(int * d_pixels, int maxRow, int maxCol, int edgeWidth) {

	int rowNum = blockIdx.y * blockDim.y + threadIdx.y;
	int colNum = blockIdx.x * blockDim.x + threadIdx.x;
	int i = rowNum * maxCol + colNum;
	
	if(colNum < maxCol && rowNum < maxRow) {
		if(colNum >= (maxCol - edgeWidth) || colNum <= edgeWidth || rowNum >= (maxRow - edgeWidth) || rowNum <= edgeWidth) {
			d_pixels[i] = 0;
		}
	}
}

__host__ void gpuCircleSolution(int ** pixels, int numRows, int numCols, int centerRow, int centerCol, int radius) {
	
	size_t num_bytes = numRows * numCols * sizeof(int);
	int * h_pixels = convertArrayToSingle(pixels, numRows, numCols); 
	int * d_pixels;
	cudaMalloc(&d_pixels, num_bytes);
	cudaMemcpy(d_pixels, h_pixels, num_bytes, cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = ceil((float) numCols / block.x);
	grid.y = ceil((float) numRows / block.y);


	//Kernel call
	//gpuEdge<<< grid, block >>>(d_pixels, numRows, numCols, edgeWidth);
	GPUCircle<<<grid, block>>>(d_pixels, centerRow, centerCol, radius, numRows, numCols);
	cudaMemcpy(h_pixels, d_pixels, num_bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);
	
	copyArrayBack(pixels, h_pixels, numRows, numCols); 
	free(h_pixels);

}

//GPU solution for the computed circle case 
__global__ void GPUCircle(int * d_pixels, int cRow, int cCol, int radius, int maxRow, int maxCol) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i = row * maxCol + col;
	int p1[2] = {cRow, cCol};
	int p2[2] = {row, col};
	
	if(row < maxRow && col < maxCol) {
		if(distance(p1, p2) < radius) {
			d_pixels[i] = 0;
		}
	}
}

__device__ float distance( int p1[], int p2[] ) {
    double xDistance = p2[0] - p1[0];
    double yDistance = p2[1] - p1[1];
    double square = xDistance * xDistance + yDistance * yDistance;
    return sqrt(square);
}

__host__ void gpuLineSolution(int ** pixels, int numRows, int numCols , int p1row, int p1col, int p2row, int p2col) {

	size_t num_bytes = numRows * numCols * sizeof(int);
	int * h_pixels = convertArrayToSingle(pixels, numRows, numCols); 
	int * d_pixels;
	cudaMalloc(&d_pixels, num_bytes);
	cudaMemcpy(d_pixels, h_pixels, num_bytes, cudaMemcpyHostToDevice);

	dim3 grid, block;
	block.x = 16;
	block.y = 16;
	grid.x = ceil((float) numCols / block.x);
	grid.y = ceil((float) numRows / block.y);


	//Kernel call
	//gpuEdge<<< grid, block >>>(d_pixels, numRows, numCols, edgeWidth);
	GPULine<<<grid, block>>>(d_pixels, numRows, numCols, p1row, p1col, p2row, p2col);
	cudaMemcpy(h_pixels, d_pixels, num_bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);
	
	copyArrayBack(pixels, h_pixels, numRows, numCols); 
	free(h_pixels);

}

//GPU solution for the line
__global__ void GPULine(int * d_pixels, int maxRow, int maxCol, int p1row, int p1col, int p2row, int p2col) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i = row * maxCol + col;
	
	double slope = 0;
	double intercept = 0;
	
	if(row < maxRow && col < maxCol) {
		if(p1col != p2col) {
			slope = calcSlope(p1row, p1col, p2row, p2col);
			intercept = calcIntercept(p1col, p1row, slope);
			if(isInLine(slope, intercept, row, col) == 0) {
				if(col <= findMax(p1col, p2col) && col >= findMin(p1col, p2col) && row <= findMax(p1row, p2row) && col >= findMin(p1row, p2row))
					d_pixels[i] = 0;
			}
		}
		else if (p1col == p2col) {
			if(col == p1col && row <= findMax(p1row, p2row) && row >= findMin(p1row, p2row)) {
				d_pixels[i] = 0;
			}
		}
	}
}

__device__ int isInLine(double slope, double intercept, int x, int y) {
	double testIntercept = y - (x * slope);
	double epsilon = .5;
	double result = testIntercept - intercept;
	
	if(abs(result) < epsilon)
		return 0;

	return 1;
}

__device__ double calcSlope(int p1row, int p1col, int p2row, int p2col) {
	double denominator = (p2col - p1col);
	double numerator = (p2row - p1col);
	
	return (denominator / numerator);
}

__device__ double calcIntercept( int col, int row, int slope) {
	return col - (row * slope);
}

__device__ int findMin(int x, int y) {
	if(x < y) 
		return x;
	
	return y;
}

__device__ int findMax(int x, int y) {
	if(x > y) 
		return x;
		
	return y;
}

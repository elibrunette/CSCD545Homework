
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] );

//GPU solution for the Edge case
__device __ void GPUEdge(d_pixels, maxRow, maxCol,edgeWidth);

//GPU solution for the Circle case 
__device__ void GPUCircle(int * d_pixels, int cRow, int cCol, int radius, int maxRow, int maxCol);

//GPU solution for the line case 
__device__ void GPULine(int * d_pixels, int maxRow, int maxCol, double slope, double intercept);

//Computes the intercept of the line given 
__device__ double calcIntercept( int col, int row, slope);
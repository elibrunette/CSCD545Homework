
__device__ void gpuKNN();

__global__ void KNN(int * inx, int * iny, int nx, int ny, double * out);
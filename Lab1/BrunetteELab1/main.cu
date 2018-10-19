#include <stdio.h>
#include <stdlib.h>
#include "timing.h"

#define MAXBLOCKSIZE = 1024;

typedef unsigned long long bignum;

__global__ void computeGPUPrimes(char * resultSet, bignum n);
__device__ __host__ int isPrime(bignum x);
__host__ void initializeArray(char a[], bignum len);
__host__ int arrSum(char a[], bignum len);
__host__ void computePrimes(char results[], bignum n);

__global__ void computeGPUPrimes(char * resultSet, bignum n) {

   bignum i = 2 * (blockIdx.x*blockDim.x+threadIdx.x) + 1;
   if(i < n)
     resultSet[i]=isPrime(i);   
}

__device__ __host__ int isPrime(bignum x) {
  bignum i;
  bignum lim = (bignum) sqrt((float) x) + 1;

  for(i = 3; i < lim; i=i+2) {
    if(x % i == 0)
      return 0;
  } 

  return 1;
}

__host__ void initializeArray(char a[], bignum len){

   int i;
   
   for(i=0; i<len; i++){
      a[i]= 0;
   }

}

__host__ int arrSum(char a[], bignum len){
    int i, s = 0;
    for( i = 0; i < len; i ++ ) 
    {
       s = s + a[i];
    }
  
    return s;
}

__host__ void computePrimes(char results[], bignum n){
   bignum i;
  
   //only check odd numbers
   for(i=0; i < n; i+=2)
   {
      results[i]=isPrime(i + 3);
   }
}



int main(int argc, const char * argv[]) {    
   
   if(argc < 3)
   {
       printf("Usage: prime upbound\n");
       exit(-1);
   }
   
   //calculating block size and num of threads
   bignum N = (bignum) atoi(argv[1]);
   int blockSize = (int) atoi(argv[2]);
   bignum gridSize = (bignum) ceil((N + 1) / 2.0 / blockSize);


   if(N <= 0)
   {
       printf("Usage: prime upbound, you input invalid upbound number!\n");
       exit(-1);
   } 

   if((blockSize < 1) || (blockSize > 1024)) {
      printf("Usage: upper-bound block-size, your block-size number is invalid!");
      exit(-1);
   }
   //double initializers.
   double nowHost, thenHost;
   double nowDevice, thenDevice;
   double scostHost, pcostDevice;
   
   //initializing vectors 
   char * h_results = (char *) calloc((N + 1), sizeof(char));
   char * d_results = (char *) calloc((N + 1), sizeof(char));
   char * d_hostResults = (char *) calloc((N + 1), sizeof(char));

   size_t bytes = (N + 1) * sizeof(char);

   h_results[0] = 1;
   h_results[1] = 1;
   d_results[0] = 1;
   d_results[1] = 1;
      
   //host test
   thenHost = currentTime();
   computePrimes(h_results, N + 1);
   nowHost = currentTime();
   scostHost = nowHost - thenHost;
   printf("%%%%%% Serial code executiontime in second is %lf\n", scostHost);
   printf("Total number of primes in that range is: %d.\n\n", arrSum(h_results, N + 1));

   //cuda test
   thenDevice = currentTime();
   cudaMalloc(&d_results, bytes);
   h_results = (char*)malloc(bytes);
   initializeArray(h_results, N + 1);
   cudaMemcpy(d_results, d_hostResults, bytes, cudaMemcpyHostToDevice);
   computeGPUPrimes<<<gridSize, blockSize>>>(d_results, N);
   cudaMemcpy(d_hostResults, d_results, bytes, cudaMemcpyDeviceToHost);
   nowDevice = currentTime();
   pcostDevice = nowDevice - thenDevice;
   printf("%%%%%% Parallel code executiontime on the GPU in second is %lf\n", pcostDevice);
   printf("Total number of primes in that range is: %d.\n\n", arrSum(d_hostResults, N + 1));
   printf("%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %lf\n", (scostHost / pcostDevice));

   cudaFree(d_results);
   free(h_results);
   free(d_hostResults);

   return 0;
}

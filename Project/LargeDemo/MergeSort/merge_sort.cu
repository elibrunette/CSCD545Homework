/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */




/*
 * Based on "Designing efficient sorting algorithms for manycore GPUs"
 * by Nadathur Satish, Mark Harris, and Michael Garland
 * http://mgarland.org/files/papers/gpusort-ipdps09.pdf
 *
 * Victor Podlozhnyuk 09/24/2009
 */


#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "mergeSort_common.h"



////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
static inline __host__ void getLastCudaError(const char* error) {
    cudaError_t code = code = cudaGetLastError();
    if (code != cudaSuccess)
        printf ("%s -- %s", error, cudaGetErrorString(code));
}
static inline __host__ __device__ int iDivUp(int a, int b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline __host__ __device__ int getSampleCount(int dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (sizeof(int) * 8)
static inline __device__ int nextPowerOfTwo(int x)
{
    return 1U << (W - __clz(x - 1));
}

template<int sortDir> static inline __device__ int binarySearchInclusive(double val, double *data, int L, int stride)
{
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

template<int sortDir> static inline __device__ int binarySearchInclusive(int val, int *data, int L, int stride)
{
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

template<int sortDir> static inline __device__ int binarySearchExclusive(double val, double *data, int L, int stride)
{
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

template<int sortDir> static inline __device__ int binarySearchExclusive(int val, int *data, int L, int stride)
{
    if (L == 0)
    {
        return 0;
    }

    int pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        int newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}




////////////////////////////////////////////////////////////////////////////////
// Bottom-level merge sort (binary search-based)
////////////////////////////////////////////////////////////////////////////////
template<int sortDir> __global__ void mergeSortSharedKernel(
    double *d_DstKey,
    int *d_DstVal,
    double *d_SrcKey,
    int *d_SrcVal,
    int arrayLength
)
{
    __shared__ double s_key[SHARED_SIZE_LIMIT];
    __shared__ int s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (int stride = 1; stride < arrayLength; stride <<= 1)
    {
        int     lPos = threadIdx.x & (stride - 1);
        double *baseKey = s_key + 2 * (threadIdx.x - lPos);
        int *baseVal = s_val + 2 * (threadIdx.x - lPos);

        __syncthreads();
        double keyA = baseKey[lPos +      0];
        int valA = baseVal[lPos +      0];
        double keyB = baseKey[lPos + stride];
        int valB = baseVal[lPos + stride];
        int posA = binarySearchExclusive<sortDir>(keyA, baseKey + stride, stride, stride) + lPos;
        int posB = binarySearchInclusive<sortDir>(keyB, baseKey +      0, stride, stride) + lPos;

        __syncthreads();
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];

    __syncthreads();
}

static void mergeSortShared(
    double *d_DstKey,
    int *d_DstVal,
    double *d_SrcKey,
    int *d_SrcVal,
    int batchSize,
    int arrayLength,
    int sortDir
)
{
    if (arrayLength < 2)
    {
        return;
    }

    assert(SHARED_SIZE_LIMIT % arrayLength == 0);
    assert(((batchSize * arrayLength) % SHARED_SIZE_LIMIT) == 0);
    int  blockCount = batchSize * (arrayLength / SHARED_SIZE_LIMIT);
    int threadCount = SHARED_SIZE_LIMIT / 2;

    if (sortDir)
    {
        mergeSortSharedKernel<1U><<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
        getLastCudaError("mergeSortShared<1><<<>>> failed\n");
    }
    else
    {
        mergeSortSharedKernel<0U><<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
        getLastCudaError("mergeSortShared<0><<<>>> failed\n");
    }
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
template<int sortDir> __global__ void generateSampleRanksKernel(
    int *d_RanksA,
    int *d_RanksB,
    double *d_SrcKey,
    int stride,
    int N,
    int threadCount
)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const int segmentElementsA = stride;
    const int segmentElementsB = umin(stride, N - segmentBase - stride);
    const int  segmentSamplesA = getSampleCount(segmentElementsA);
    const int  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive<sortDir>(
                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,
                          segmentElementsB, nextPowerOfTwo(segmentElementsB)
                      );
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<sortDir>(
                                                     d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,
                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA)
                                                 );
    }
}

static void generateSampleRanks(
    int *d_RanksA,
    int *d_RanksB,
    double *d_SrcKey,
    int stride,
    int N,
    int sortDir
)
{
    int lastSegmentElements = N % (2 * stride);
    int         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    if (sortDir)
    {
        generateSampleRanksKernel<1U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        getLastCudaError("generateSampleRanksKernel<1U><<<>>> failed\n");
    }
    else
    {
        generateSampleRanksKernel<0U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        getLastCudaError("generateSampleRanksKernel<0U><<<>>> failed\n");
    }
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
__global__ void mergeRanksAndIndicesKernel(
    int *d_Limits,
    int *d_Ranks,
    int stride,
    int N,
    int threadCount
)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_Ranks  += (pos - i) * 2;
    d_Limits += (pos - i) * 2;

    const int segmentElementsA = stride;
    const int segmentElementsB = umin(stride, N - segmentBase - stride);
    const int  segmentSamplesA = getSampleCount(segmentElementsA);
    const int  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        int dstPos = binarySearchExclusive<1U>(d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
        d_Limits[dstPos] = d_Ranks[i];
    }

    if (i < segmentSamplesB)
    {
        int dstPos = binarySearchInclusive<1U>(d_Ranks[segmentSamplesA + i], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
    }
}

static void mergeRanksAndIndices(
    int *d_LimitsA,
    int *d_LimitsB,
    int *d_RanksA,
    int *d_RanksB,
    int stride,
    int N
)
{
    int lastSegmentElements = N % (2 * stride);
    int         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsA,
        d_RanksA,
        stride,
        N,
        threadCount
    );
    getLastCudaError("mergeRanksAndIndicesKernel(A)<<<>>> failed\n");

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsB,
        d_RanksB,
        stride,
        N,
        threadCount
    );
    getLastCudaError("mergeRanksAndIndicesKernel(B)<<<>>> failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals
////////////////////////////////////////////////////////////////////////////////
template<int sortDir> inline __device__ void merge(
    double *dstKey,
    int *dstVal,
    double *srcAKey,
    int *srcAVal,
    double *srcBKey,
    int *srcBVal,
    int lenA,
    int nPowTwoLenA,
    int lenB,
    int nPowTwoLenB
)
{
    double keyA, keyB;
    int valA, valB, dstPosA, dstPosB;

    if (threadIdx.x < lenA)
    {
        keyA = srcAKey[threadIdx.x];
        valA = srcAVal[threadIdx.x];
        dstPosA = binarySearchExclusive<sortDir>(keyA, srcBKey, lenB, nPowTwoLenB) + threadIdx.x;
    }

    if (threadIdx.x < lenB)
    {
        keyB = srcBKey[threadIdx.x];
        valB = srcBVal[threadIdx.x];
        dstPosB = binarySearchInclusive<sortDir>(keyB, srcAKey, lenA, nPowTwoLenA) + threadIdx.x;
    }

    __syncthreads();

    if (threadIdx.x < lenA)
    {
        dstKey[dstPosA] = keyA;
        dstVal[dstPosA] = valA;
    }

    if (threadIdx.x < lenB)
    {
        dstKey[dstPosB] = keyB;
        dstVal[dstPosB] = valB;
    }
}

template<int sortDir> __global__ void mergeElementaryIntervalsKernel(
    double *d_DstKey,
    int *d_DstVal,
    double *d_SrcKey,
    int *d_SrcVal,
    int *d_LimitsA,
    int *d_LimitsB,
    int stride,
    int N
)
{
    __shared__ double s_key[2 * SAMPLE_STRIDE];
    __shared__ int s_val[2 * SAMPLE_STRIDE];

    const int   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const int segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (threadIdx.x == 0)
    {
        int segmentElementsA = stride;
        int segmentElementsB = umin(stride, N - segmentBase - stride);
        int  segmentSamplesA = getSampleCount(segmentElementsA);
        int  segmentSamplesB = getSampleCount(segmentElementsB);
        int   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        int endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        int endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }

    //Load main input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    //Merge data in shared memory
    __syncthreads();
    merge<sortDir>(
        s_key,
        s_val,
        s_key + 0,
        s_val + 0,
        s_key + SAMPLE_STRIDE,
        s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE
    );

    //Store merged data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}

static void mergeElementaryIntervals(
    double *d_DstKey,
    int *d_DstVal,
    double *d_SrcKey,
    int *d_SrcVal,
    int *d_LimitsA,
    int *d_LimitsB,
    int stride,
    int N,
    int sortDir
)
{
    int lastSegmentElements = N % (2 * stride);
    int          mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

    if (sortDir)
    {
        mergeElementaryIntervalsKernel<1U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_DstVal,
            d_SrcKey,
            d_SrcVal,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        getLastCudaError("mergeElementaryIntervalsKernel<1> failed\n");
    }
    else
    {
        mergeElementaryIntervalsKernel<0U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_DstVal,
            d_SrcKey,
            d_SrcVal,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        getLastCudaError("mergeElementaryIntervalsKernel<0> failed\n");
    }
}

static int *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
static const int MAX_SAMPLE_COUNT = 32768 * 16; //changed by Tony

extern "C" void initMergeSort(void)
{
    cudaMalloc((void **)&d_RanksA,  MAX_SAMPLE_COUNT * sizeof(int));
    cudaMalloc((void **)&d_RanksB,  MAX_SAMPLE_COUNT * sizeof(int));
    cudaMalloc((void **)&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(int));
    cudaMalloc((void **)&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(int));
}

extern "C" void closeMergeSort(void)
{
    cudaFree(d_RanksA);
    cudaFree(d_RanksB);
    cudaFree(d_LimitsB);
    cudaFree(d_LimitsA);
}

extern "C" void mergeSort(
    double *d_DstKey,
    int *d_DstVal,
    double *d_BufKey,
    int *d_BufVal,
    double *d_SrcKey,
    int *d_SrcVal,
    int N,
    int sortDir
)
{
    int stageCount = 0;

    for (int stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++);

    double *ikey, *okey;
    int *ival, *oval;

    if (stageCount & 1)
    {
        ikey = d_BufKey;
        ival = d_BufVal;
        okey = d_DstKey;
        oval = d_DstVal;
    }
    else
    {
        ikey = d_DstKey;
        ival = d_DstVal;
        okey = d_BufKey;
        oval = d_BufVal;
    }

    assert(N <= (SAMPLE_STRIDE * MAX_SAMPLE_COUNT));
    assert(N % SHARED_SIZE_LIMIT == 0);
    mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, ceil((double)N / SHARED_SIZE_LIMIT), SHARED_SIZE_LIMIT, sortDir);

    for (int stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1)
    {
        int lastSegmentElements = N % (2 * stride);

        //Find sample ranks and prepare for limiters merge
        generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, N, sortDir);

        //Merge ranks and indices
        mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, N);

        //Merge elementary intervals
        mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, N, sortDir);

        if (lastSegmentElements <= stride)
        {
            //Last merge segment consists of a single array which just needs to be passed through
            cudaMemcpy(okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements), lastSegmentElements * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(oval + (N - lastSegmentElements), ival + (N - lastSegmentElements), lastSegmentElements * sizeof(int), cudaMemcpyDeviceToDevice);
        }

        int *tval;
        double *tkey;
        tkey = ikey;
        ikey = okey;
        okey = tkey;
        tval = ival;
        ival = oval;
        oval = tval;
    }
}


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



////////////////////////////////////////////////////////////////////////////////
// Shortcut definitions
////////////////////////////////////////////////////////////////////////////////

#define SHARED_SIZE_LIMIT 1024
#define     SAMPLE_STRIDE 128



////////////////////////////////////////////////////////////////////////////////
// Extensive sort validation routine
////////////////////////////////////////////////////////////////////////////////
extern "C" int validateSortedKeys(
    double *resKey,
    double *srcKey,
    int batchSize,
    int arrayLength,
    int numValues,
    int sortDir
);

extern "C" void fillValues(
    int *val,
    int N
);

extern "C" int validateSortedValues(
    double *resKey,
    int *resVal,
    double *srcKey,
    int batchSize,
    int arrayLength
);



////////////////////////////////////////////////////////////////////////////////
// CUDA merge sort
////////////////////////////////////////////////////////////////////////////////
extern "C" void initMergeSort(void);

extern "C" void closeMergeSort(void);

extern "C" void mergeSort(
    double *dstKey,
    int *dstVal,
    double *bufKey,
    int *bufVal,
    double *srcKey,
    int *srcVal,
    int N,
    int sortDir
);



////////////////////////////////////////////////////////////////////////////////
// CPU "emulation"
////////////////////////////////////////////////////////////////////////////////
extern "C" void mergeSortHost(
    double *dstKey,
    int *dstVal,
    double *bufKey,
    int *bufVal,
    double *srcKey,
    int *srcVal,
    int N,
    int sortDir
);

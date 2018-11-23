#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "gpuKNN.h"
#include "FileUtils.h"

int main(int argc, char * argv);
void usage();
void freeIntDoublePointer(int ** input, int rows);
void print2DArray(int ** arr, int col, int row);
void printSingleArray(double * arr, int n);
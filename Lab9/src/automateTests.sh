#!/bin/sh
# This is a comment!
make
#./mmul blockWidth matrixFileName p
#serial solutions 

make
./mmul 8 ../data/1024.mat
diff ./gpuout2 ./gpuout3

./mmul 16 ../data/1024.mat
diff ./gpuout2 ./gpuout3

./mmul 8 ../data/2048.mat
diff ./gpuout2 ./gpuout3

./mmul 16 ../data/2048.mat
diff ./gpuout2 ./gpuout3

./mmul 32 ../data/2048.mat
diff ./gpuout2 ./gpuout3
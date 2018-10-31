#!/bin/sh
# This is a comment!
make
#./jacobi threadsPerBlock passes width height [p]
#serial solutions 

./jacobi 0 1000 1600 1600
./jacobi 0 1000 3200 3200

#gpu solution 1600 by 1600
./jacobi 32 1000 1600 1600
./jacobi 128 1000 1600 1600
./jacobi 256 1000 1600 1600
./jacobi 384 1000 1600 1600
./jacobi 512 1000 1600 1600

#gpu solution 3200 by 3200
./jacobi 1024 1000 1600 1600
./jacobi 256 1000 3200 3200

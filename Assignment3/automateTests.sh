#!/bin/sh
# This is a comment!
make
#./jacobi threadsPerBlock passes width height [p]
#serial solutions 

cd ./k1
make dbg="" cuda_dbg="" opt="-O3"
./jacobi 512 1000 1600 1600 p > k21_output

cd ../k0
make dbg="" cuda_dbg="" opt="-O3"
./jacobi 512 1000 1600 1600 p > k20_output

cd ..
diff ./k0/k20_output ./k1/k21_output
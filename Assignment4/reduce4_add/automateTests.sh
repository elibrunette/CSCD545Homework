#!/bin/sh
# This is a comment!
#./reduce blockWidth numElementsInput p
#serial solutions 

cd ./k2
make
./jacobi 16 50 p > k21_output

cd ../k1
make
./jacobi 16 50 p > k20_output

cd ..
diff ./k0/k20_output ./k1/k21_output
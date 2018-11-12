#!/bin/sh
# This is a comment!
#./reduce blockWidth numElementsInput p

make
./reduce 1024 1048576
./reduce 1024 16777216
./reduce 1024 67108864
./reduce 1024 134217728
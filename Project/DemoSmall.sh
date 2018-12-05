#!/bin/sh
# This is a comment!

cd ./SmallDemo

./knn  3 5 2 20 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

cd ..


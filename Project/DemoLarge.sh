#!/bin/sh
# This is a comment!

cd ./LargeDemo

./knn  3 500 100 10000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput


cd ..
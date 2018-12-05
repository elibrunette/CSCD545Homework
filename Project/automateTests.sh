#!/bin/sh
# This is a comment!

cd ./LargeDemo

make
./knn  3 5 2 20 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput


./knn  3 500 100 200 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

./knn  3 500 100 1000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

./knn  3 500 100 5000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

./knn  3 500 100 10000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

./knn  3 500 100 20000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

./knn  3 500 100 30000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

./knn  3 500 100 50000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput

./knn  3 500 100 70000 cpuOutput gpuOutput
diff ./cpuDistance ./gpuDistance
diff ./cpuOutput ./gpuOutput



./knn 3 50 
cd ..



knn: gpuKNN.o knn.o FileUtils.o ArrayUtils.o cpuKNN.o mergeSortInterface.o timing.o
	nvcc -arch=sm_30 -o knn gpuKNN.o knn.o FileUtils.o ArrayUtils.o cpuKNN.o timing.o mergeSortInterface.o -I.

gpuKNN.o: gpuKNN.cu
	nvcc -arch=sm_30 -c gpuKNN.cu

cpuKNN.o: cpuKNN.c cpuKNN.h
	g++ -c -x c++ cpuKNN.c -I.

mergeSortInterface.o: mergeSortInterface.cu
	nvcc -arch=sm_30 -c mergeSortInterface.cu

knn.o: knn.c knn.h
	g++ -c -x c++ knn.c -I. 

FileUtils.o: FileUtils.c FileUtils.h
	g++ -c -x c++ FileUtils.c -I.

ArrayUtils.o: ArrayUtils.c ArrayUtils.h
	g++ -c -x c++ ArrayUtils.c -I.

timing.o: ./Timing/timing.c
	g++ -c -x c++ ./Timing/timing.c -I.


clean:
	rm -r *.o knn

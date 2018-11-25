knn: gpuKNN.o knn.o FileUtils.o ArrayUtils.o
	nvcc -arch=sm_30 -o knn gpuKNN.o knn.o FileUtils.o ArrayUtils.o -I.

gpuKNN.o: gpuKNN.cu
	nvcc -arch=sm_30 -c gpuKNN.cu

knn.o: knn.c knn.h
	g++ -c -x c++ knn.c -I. 

FileUtils.o: FileUtils.c FileUtils.h
	g++ -c -x c++ FileUtils.c -I.

ArrayUtils.o: ArrayUtils.c ArrayUtils.h
	g++ -c -x c++ ArrayUtils.c -I.

clean:
	rm -r *.o knn

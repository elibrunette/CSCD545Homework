knn: gpuKNN.o knn.o FileUtils.o
	nvcc -arch=sm_30 -o knn gpuKNN.o knn.o FileUtils.o -I.

gpuKNN.o: gpuKNN.cu gpuKNN.h
	nvcc -arch=sm_30 -c gpuKNN.cu

knn.o: knn.c knn.h
	g++ -c -x c++ knn.c -I. 

FileUtils.o: FileUtils.c FileUtils.h
	g++ -c -x c++ FileUtils.c -I.

clean:
	rm -r *.o knn

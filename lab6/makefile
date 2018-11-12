lab6: lab6.o
	nvcc -arch=sm_30 -o lab6 lab6.o

lab6.o: lab6.cu
	nvcc -g -arch=sm_30 -c lab6.cu

clean:
	rm -r *.o lab6

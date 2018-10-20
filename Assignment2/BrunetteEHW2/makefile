myPaint: pgmProcess.o pgmUtility.o main.o timing.o
	nvcc -arch=sm_30 -o myPaint pgmProcess.o pgmUtility.o main.o timing.o -I.

pgmProcess.o: pgmProcess.cu
	nvcc -arch=sm_30 -c pgmProcess.cu

pgmUtility.o: pgmUtility.c pgmUtility.h
	g++ -c -o pgmUtility.o pgmUtility.c -I.

timing.o: timing.c timing.h
	g++ -c -x c++ timing.c -I.

main.o: main.cu
	nvcc -arch=sm_30 -c main.cu 

clean:
	rm -r *.o myPaint

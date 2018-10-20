#!/bin/sh
# This is a comment!
make
./myPaint -c 470 355 100 ./balloons.ascii.pgm balloons_c100_4.pgm
./myPaint -c 228 285 75 ./balloons.ascii.pgm balloons_c75_5.pgm 
./myPaint -e 50 ./balloons.ascii.pgm balloons_e50_2.pgm
./myPaint -l 1 5 50 200 ./balloons.ascii.pgm balloons_l1.pgm
./myPaint -l 1 50 479 639 ./balloons.ascii.pgm balloons_l2.pgm
./myPaint -l 479 5 0 639 ./balloons.ascii.pgm balloons_l3.pgm
./myPaint -l 5 320 240 320 ./balloons.ascii.pgm balloons_l4.pgm
#./myPaint -ce 470 355 100 50 ./balloons.ascii.pgm balloons_c100e50_4.pgm
#./myPaint -c -e 228 285 75 100 ./balloons.ascii.pgm balloons_c75_e100.pgm 

#./myPaint -c 470 355 100 ./blankCanvas1.ascii.pgm blankCanvas1_c100_4.pgm | blankCanvasResults1.txt
#./myPaint -c 228 285 75 ./blankCanvas1.ascii.pgm blankCanvas1_c75_5.pgm | blankCanvasResults1.txt
#./myPaint -e 50 ./blankCanvas1.ascii.pgm blankCanvas1_e50_2.pgm > blankCanvasResults1.txt
#./myPaint -l 1 5 50 200 ./blankCanvas1.ascii.pgm blankCanvas1_l1.pgm > blankCanvasResults1.txt
#./myPaint -l 1 50 479 639 ./blankCanvas1.ascii.pgm blankCanvas1_l2.pgm > blankCanvasResults1.txt
#./myPaint -l 479 5 0 639 ./blankCanvas1.ascii.pgm blankCanvas1_l3.pgm > blankCanvasResults1.txt
#./myPaint -l 5 320 240 320 ./blankCanvas1.ascii.pgm blankCanvas1_l4.pgm > blankCanvasResults1.txt

./myPaint -c 470 355 100 ./blankCanvas2.ascii.pgm blankCanvas2_c100_4.pgm > blankCanvasResults2.txt
#./myPaint -c 228 285 75 ./blankCanvas2.ascii.pgm blankCanvas2_c75_5.pgm > blankCanvasResults2.txt
#./myPaint -e 50 ./blankCanvas1.asci2.pgm blankCanvas2_e50_2.pgm > blankCanvasResults2.txt
#./myPaint -l 1 5 50 200 ./blankCanvas2.ascii.pgm blankCanvas2_l1.pgm > blankCanvasResults2.txt
#./myPaint -l 1 50 479 639 ./blankCanvas2.ascii.pgm blankCanvas2_l2.pgm > blankCanvasResults2.txt
#./myPaint -l 479 5 0 639 ./blankCanvas2.ascii.pgm blankCanvas2_l3.pgm > blankCanvasResults2.txt
#./myPaint -l 5 320 240 320 ./blankCanvas2.ascii.pgm blankCanvas2_l4.pgm > blankCanvasResults2.txt

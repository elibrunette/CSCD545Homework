#include <stdio.h>
#include <stdlib.h>


int main(int argc, char * argv[]) {
	if(argc == 5) {
		//printf("made it\n\n\n");
		char * sizeRow = argv[1];
		char * sizeCol = argv[2];
		char * fileName = argv[4];
		char * maxIntensity = argv[3];
		int maxRow = atoi(sizeRow);
		int maxCol = atoi(sizeCol);
		int intensity = atoi(maxIntensity);
		int row = 0;
		int col = 0;
		int counter = 1; 

		FILE * out = fopen(fileName, "w");

		fputs("P2\n",out);
		fputs("#Generic large text file\n", out);
		fprintf(out, "%d %d\n", col, row);
		fprintf(out, "%d\n", intensity);
		
		for(row = 0; row < maxRow; row++) {
			for(col = 0; col < maxCol; col++) {
				if(counter < 14)
					fprintf(out, "%d ", intensity);
				else {
					counter = 0; 
					fprintf(out, "%d\n", intensity);
				}

				counter ++;
			}
			counter = 0;
			fprintf(out, "\n");
		}
		fclose(out);
		printf("done\n");
	}
}

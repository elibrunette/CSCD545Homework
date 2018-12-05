#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int * getHeader(FILE * fin) {
	int * toReturn = (int *) calloc(2, sizeof(int));
	char * str1;
	char * str2;
	int row;
	int col;
	int count = 0;
	char * token;
	char * remainder;
	char buff[255];
	//printf("Made it to getHeader\n\n\n");
	if(fin == NULL) {
		printf("fin is null\n\n\n");
		return NULL;
	}
	if(fgets(buff, 255, fin) != NULL) {
		//printf("Read in line: %s", buff);
		remainder = buff;
		while(token = strtok_r(remainder, " ", &remainder)) {
			if(strcmp(token, "\n") != 0) {
				toReturn[count++] = atoi(token);
			}
		}
	}
	//printf("toReturn[0] = %d\ntoReturn[1] = %d\n", toReturn[0], toReturn[1]);
	return toReturn;
}

int ** readFile(FILE * fin, int col, int row) {

	//printf("row: %d\ncol: %d\n\n\n", row, col);
	char buff[255];
	int i = 0;
	char * token;
	char * remainder;
	int toAdd = 0;
	int counter = 0;
	int colCount = 0;
	int rowCount = 0;

	int ** toReturn = (int **) calloc((row) + 1, sizeof(int *));

	for(counter = 0; counter < row; counter ++) {
		toReturn[counter] = (int *) calloc((col) + 1, sizeof(int));
	}
	//printf("Finished Initializing toReturn array\n\n\n");

	while( fgets(buff, 255, fin) != NULL) {
		//printf("Line read in: %s\n", buff);
		remainder = buff;
		while(token = strtok_r(remainder, " ", &remainder)) {
			//printf("token: %s", token);
			if(strcmp(token, "\n") != 0) {
				toAdd = atoi(token);
				toReturn[rowCount][colCount] = toAdd;
				colCount++;
			}
		}
		rowCount++;
		colCount = 0;
		//printf("newLine\n");
	}
	//printf("finished initializing the array\n\n\n");
	return toReturn;
}

void outputToFile(const char * fileName, double ** arr, int row, int col) {
	FILE * fout = fopen(fileName, "w");
	int x = 0; 
	int y = 0;

	//account for edge cases:
		//turnary operator for end of row case and end case 
	for(x = 0; x < row; x++) {
		for(y = 0; y < col; y++) {
			fprintf(fout, "%f ", arr[x][y]);
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

void outputFinal(const char * fileName, int ** testPoints, int * classifiedPoints, int testCol, int testRow, double time) {
	FILE * fout = fopen(fileName, "w");
	int x = 0; 
	int y = 0; 
	for(x = 0; x < testRow; x++) {
		for(y = 0; y < testCol; y++) {
			fprintf(fout, "%d ", testPoints[x][y]);
		}
		fprintf(fout, "%d\n", classifiedPoints[x]);
	}
	fprintf(fout, "%f", time);

}
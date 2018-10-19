#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "pgmUtility.h"
#include "pgmProcess.h"

char ** processHeader(FILE * fin);
int * getDementionsOfPicture(const char ** header);
void freePointers(char ** header, int ** processedPicture, int numRows);
void printPicture(int ** processedPicture, int numRows, int numCols);
void printHeader(char ** header);


int main(int argc, char * argv[] )
{
	//printf("made it into main\n\n\n");
	//variables for processing
	int i = 2;
	char * inputImage;
	char * outputImage;
	int edgeWidth = -1;
	int circleCenterRow = -1;
	int circleCenterColumn = -1;
	int radius = -1;
	int p1row = -1;
	int p1col = -1;
	int p2row = -1;
	int p2col = -1;
	int * dementions;
	int numRows = 0;
	int numCols = 0;

	char ** header;
	int ** processedPicture;

	//printf("made it to before the switch statement\n\n\n");
	if( ( argc > 1) && ( argv[1][0] == '-' ) ) {
		//printf("made it into the swtich statement\n\n\n");
		switch ( argv[1][1] ) {

			//edge
			case 'e':				
				if( !argv[i] || argc < 5 || argc > 9 ) {
					printf("Please enter a valid number of arguments");
					break;
				}

				if( argc == 5) {
					edgeWidth = atoi(argv[2]);
					outputImage = argv[4];
					inputImage = argv[3];
					if(edgeWidth != 0) {
						//TODO
						//process input file
						FILE * fin = fopen(inputImage,"r");

						header = processHeader(fin);
						//printHeader(header);
						dementions = getDementionsOfPicture((const char **) header);
						//printHeader(header);
						numRows = dementions[1];
						numCols = dementions[0];

						processedPicture = pgmRead(header, &numRows, &numCols, fin);

						//printPicture(processedPicture, numRows, numCols);
						//process edge
						pgmDrawEdge(processedPicture, numRows, numCols, edgeWidth, header);
						fclose(fin);
						//output image
						FILE * fout = fopen(outputImage, "w");
						//printHeader(header);
						if(pgmWrite((const char ** )header, (const int **) processedPicture, numRows, numCols, fout) == -1) {
							printf("Could not write to file");
						}
							
					} 
				}
				else if( argc == 9) {
					circleCenterRow = atoi(argv[3]);
					circleCenterColumn = atoi(argv[4]);
					radius = atoi(argv[5]);
					edgeWidth = atoi(argv[6]);
					inputImage = argv[7];
					outputImage = argv[8];
					//TODO
					//process input file
					//process circle and edge
				} else {
					printf("Please enter a valid amount of arguments");
				}
               			break;

			//circle
			case 'c':
				if( !argv[i] || argc < 7 || argc > 8) {
					printf("Please enter a valid number of arguments.");
					break;
				}
				if( argc == 7) {
					circleCenterRow = atoi(argv[2]);
					circleCenterColumn = atoi(argv[3]);
					radius = atoi(argv[4]);
					inputImage = argv[5];
					outputImage = argv[6];
					
					//process input file
					FILE * fin = fopen(inputImage,"r");
					
					//Read in the header
					header = processHeader(fin);
					dementions = getDementionsOfPicture((const char **) header);
					numRows = dementions[1];
					numCols = dementions[0];

					//read in the picture to store
					processedPicture = pgmRead(header, &numRows, &numCols, fin);
					fclose(fin);

					//process the picture
					pgmDrawCircle( processedPicture, numRows, numCols, circleCenterRow, circleCenterColumn, radius, header );

					//output image
					FILE * fout = fopen(outputImage, "w");
					if(pgmWrite((const char ** )header, (const int **) processedPicture, numRows, numCols, fout) == -1) {
						printf("Could not write to file");
					}
				}
				else if( (strcmp(argv[1], "-ce") == 0) && argc == 8 ) {
					circleCenterRow = atoi(argv[2]);
					circleCenterColumn = atoi(argv[3]);
					radius = atoi(argv[4]);
					edgeWidth = atoi(argv[5]);
					inputImage = argv[6];
					outputImage = argv[7];
					
					//process input file
					//process circle and edge
				}
               			break;

			//line
			case 'l':
				//printf("made it into the line command");
				if( !argv[i] || argc < 8 || argc > 8) {
					//printf("Please enter a valid amount of arguments");
					break;
				}
				if(argc == 8) {
					//printf("made it into the line command");
					p1row = atoi(argv[2]);
					p1col = atoi(argv[3]);
					p2row = atoi(argv[4]);
					p2col = atoi(argv[5]);
					inputImage = argv[6];
					outputImage = argv[7];

					//process input file
					FILE * fin = fopen(inputImage,"r");
					//printf("made it to before reading in the header\n\n\n\n");

					//Read in the header
					header = processHeader(fin);
					dementions = getDementionsOfPicture((const char **) header);
					numRows = dementions[1];
					numCols = dementions[0];
					//printf("made it after reading in header and getting the dementions\n\n\n");

					//read in the picture to store
					processedPicture = pgmRead(header, &numRows, &numCols, fin);
					fclose(fin);
					//printf("made it to after reading in the picture\n\n\n");

					//process the picture
					pgmDrawLine( processedPicture, numRows, numCols, header, p1row, p1col, p2row, p2col );

					//output image
					FILE * fout = fopen(outputImage, "w");
					if(pgmWrite((const char ** )header, (const int **) processedPicture, numRows, numCols, fout) == -1) {
						printf("Could not write to file");
					}					
				}
               			break;
	   		default:
				printf("Wrong flag: %s\n", argv[1]);
			
		}
	}

	
	//free double pointer header
	//free dementions array
	freePointers(header, processedPicture, numRows);	
	//double pointer processed file
	return 0;
}


char ** processHeader(FILE * fin) {
	char ** toReturn = (char **) calloc(4, sizeof(char *));
	char buffer[200];
	int counter = 0;

	//initialize the double array
	for(counter = 0; counter < 4; counter++) {
		toReturn[counter] = (char *) calloc(200, sizeof(char));
	}

	//copy the char array
	for(counter = 0; counter < 4; counter++) {
		if(fgets(buffer, 200, fin) != NULL) {
			strcpy(toReturn[counter], buffer);
		}
	}

	return toReturn;
}

int * getDementionsOfPicture(const char ** header) {
	int * toReturn = (int *) calloc( 2, sizeof(int));
	char * parsedHeader = (char *) calloc(100, sizeof(char));
	strcpy(parsedHeader, header[2]);
	char * token;
	char * remainder = parsedHeader;

	token = strtok_r(remainder, " ", &remainder);
	toReturn[0] = atoi(token);
	token = strtok_r(remainder, " ", &remainder);
	toReturn[1] = atoi(token);
	//free(parsedHeader);
	return toReturn; 
}

void freePointers(char ** header, int ** processedPicture, int numRows) {
	int counter = 0; 

	for(counter = 0; counter < 4; counter++) {
		free(header[counter]);
	}
	free(header);

	for(counter = 0; counter < numRows; counter ++) {
		free(processedPicture[counter]);
	}

	free(processedPicture);
}

void printPicture(int  ** processedPicture, int numRows, int numCols) {
	int i = 0; 
	int j = 0;

	for(i = 0; i < numRows; i++) {
		for(j = 0; j < numCols; j++) {
			printf("%d ", processedPicture[i][j]);
		}
	}
}

void printHeader(char ** header) {	
	printf("%s", header[0]);
	printf("%s", header[1]);
	printf("%s", header[2]);
	printf("%s", header[3]);
}
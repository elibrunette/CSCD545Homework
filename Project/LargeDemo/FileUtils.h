int ** readFile(FILE * fin, int col, int row);
int * getHeader(FILE * fin);
void outputToFile(const char * fileName, double ** arr, int row, int col);
void outputFinal(const char * fileName, int ** testPoints, int * classifiedPoints, int testCol, int testRow, double time);
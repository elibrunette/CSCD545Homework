void gpu_merge_sort(double * h_dist, int * h_label, int n_labeled, int n_test_cases) {
	/*
		h_dist: size n_test_cases * n_labeled, each segment of length n_labeled has all the distances for a test case
		h_label: size n_test_cases * n_labeled, each segment of length n_labeled has all the labels for a test case
		n_labeled: the number of labeled points in each test case
		n_test_cases: number of test cases
	*/
    double  *d_SrcKey, *d_BufKey, *d_DstKey, *h_dist_temp;
    int     *d_SrcVal, *d_BufVal, *d_DstVal, *h_label_temp;

    int N_sort = pow(2, ceil(log(n_labeled)/log(2)));                      //size of array to sort, with padding (gets the next power of 2)

    cudaMalloc((void **)&d_SrcKey, N_sort * sizeof(double));
    cudaMalloc((void **)&d_DstKey, N_sort * sizeof(double));
    cudaMalloc((void **)&d_BufKey, N_sort * sizeof(double));
    cudaMalloc((void **)&d_SrcVal, N_sort * sizeof(int));
    cudaMalloc((void **)&d_DstVal, N_sort * sizeof(int));
    cudaMalloc((void **)&d_BufVal, N_sort * sizeof(int));

    initMergeSort(); //included in mergeSort_common.h			
	
	h_dist_temp = (double*)malloc(N_sort * sizeof(double));         //allocate memory for padded distances array
    h_label_temp  = (int*)   malloc(N_sort * sizeof(int));            //allocate memory for padded values array

    for(int i = 0; i < n_test_cases; i++) {                                    //run for each test case (n_test_cases = number of unlabeled points)

        for(int j = 0; j < N_sort; j++) {                                  //pad temp arrays
			if(j < n_p) {
				h_dist_temp[j] = h_dist[j];
				h_label_temp[j] = h_label[j];
			}
			else {
				h_dist_temp[j] = INFINITY;
				h_label_temp[j] = 0;
			}
        }

        cudaMemcpy(d_SrcKey, h_dist_temp, N_sort * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_SrcVal, h_label_temp, N_sort * sizeof(int), cudaMemcpyHostToDevice);

        mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N_sort, 1);

        cudaMemcpy(h_dist, d_DstKey, n_labeled * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_label, d_DstVal, n_labeled * sizeof(int), cudaMemcpyDeviceToHost);

        //do whatever before moving on to the next test case
    }

    closeMergeSort();

    cudaFree(d_SrcKey);
    cudaFree(d_DstKey);
    cudaFree(d_BufKey);
    cudaFree(d_DstVal);
    cudaFree(d_SrcVal);
    cudaFree(d_BufVal);

    free(h_dist_temp);
    free(h_label_temp);
}
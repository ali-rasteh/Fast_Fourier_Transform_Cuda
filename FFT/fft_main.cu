//Do NOT MODIFY THIS FILE

#include "fft.h"

// ===========================> Functions Prototype <===============================
void fill(float* data, int size);
double calc_mse(float* data1_r, float* data1_i, float* data2_r, float* data2_i, int size);
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& M, unsigned int& SIMPLE);
void cpuKernel(float* X_serial_r, float* X_serial_i, int n, float* tmp_r, float* tmp_i);
void gpuKernels(float* x_r, float* x_i, float* X_r, float* X_i, unsigned int N, unsigned int M, unsigned int SIMPLE, double* gpu_kernel_time);
// =================================================================================





/**
 * @file fft_main.cu
 * @brief Main program for performing Fast Fourier Transform (FFT) using CUDA.
 *
 * This program performs FFT on a given set of data using both CPU and GPU, and compares their performance.
 *
 * @param argc Number of command line arguments.
 * @param argv Array of command line arguments.
 *
 * The command line arguments are used to get the parameters for the FFT calculation.
 *
 * The program performs the following steps:
 * 1. Retrieves and prints the CUDA device properties.
 * 2. Gets the parameters from the command line.
 * 3. Allocates memory on the CPU for the input and output data.
 * 4. Fills the input arrays with random values.
 * 5. Performs FFT on the CPU and measures the time taken.
 * 6. Performs FFT on the GPU and measures the time taken.
 * 7. Calculates the mean squared error (MSE) between the CPU and GPU results.
 * 8. Prints the results including the time taken by CPU and GPU, and the MSE.
 * 9. Frees the allocated memory.
 *
 * @note The program contains commented-out code for debugging and verification purposes.
 */
int main(int argc, char *argv[]) {


    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);

    // get parameters from command line
    unsigned int N, M, SIMPLE;
    get_inputs(argc, argv, N, M, SIMPLE);

    // allocate memory in CPU for calculation
    float* x_r; // real part
    float* x_i; // imaginary part
    float* X_serial_r;
    float* X_serial_i;
    float* X_r;
    float* X_i;
    x_r = (float*) malloc(N * sizeof(float));
    x_i = (float*) malloc(N * sizeof(float));
    X_serial_r = (float*) malloc(N * sizeof(float));
    X_serial_i = (float*) malloc(N * sizeof(float));
    X_r = (float*) malloc(N * sizeof(float));
    X_i = (float*) malloc(N * sizeof(float));

    // fill x_r and x_i arrays with random values between -8.0f and 8.0f
    srand(0);
    fill(x_r, N);
    fill(x_i, N);
	int i; for (i = 0; i < N; i++) {
		X_serial_r[i] = x_r[i];
		X_serial_i[i] = x_i[i];
	}

    // time measurement for CPU calculation
	float *tmp_r, *tmp_i;
	tmp_r = (float*) malloc(N * sizeof(float));
    tmp_i = (float*) malloc(N * sizeof(float));
    clock_t t0 = clock();
    cpuKernel(X_serial_r, X_serial_i, N, tmp_r, tmp_i);
    clock_t t1 = clock();
	free(tmp_r); free(tmp_i);

    // time measurement for GPU calculation
	double gpu_kernel_time = 0.0;
    clock_t t2 = clock();
	gpuKernels(x_r, x_i, X_r, X_i, N, M, SIMPLE, &gpu_kernel_time);
    clock_t t3 = clock();

    // check correctness of calculation
    double mse = calc_mse(X_serial_r, X_serial_i, X_r, X_i, N);
	printf("simple=%d m=%d n=%d CPU=%g ms GPU=%g ms GPU-Kernels=%g ms mse=%g\n",
	SIMPLE, M, N, (t1-t0)/1000.0, (t3-t2)/1000.0, gpu_kernel_time, mse);
	
	/*
	for (i = 0; i<N; i++) {
		printf("%f\t%f\n", x_r[i], x_i[i]);
	}
	printf("\n");
	for (i = 0; i<N; i++) {
		printf("%f\t%f\n", X_serial_r[i], X_serial_i[i]);
	}
	*/
	
	
	
	
	
	/*
	printf("\n");
	for (i = 0; i<N; i++) {
		printf("%d:\t%f\t%f\n", i, X_serial_r[i], X_serial_i[i]);
	}
	printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n middle \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	for (i = 0; i<N; i++) {
		printf("%d:\t%f\t%f\n", i, X_r[i], X_i[i]);
	}
	printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n end \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	*/
	/*
	for (i = 0; i<N; i++) {
		if( ((X_serial_r[i] - X_r[i] > 0.01) || (X_r[i] - X_serial_r[i] > 0.01)) || ((X_serial_i[i] - X_i[i] > 0.01) || (X_i[i] - X_serial_i[i] > 0.01)) ){
			printf("%d\n", i);
		}
		/*if((X_serial_i[i] - X_i[i] < 0.01) || (X_i[i] - X_serial_i[i] < 0.01)){
			printf("image %d\n", i);
		}
	}
	*/
	/*
	for (i = 0; i<100; i++) {
		printf("%d:\t%f\t%f\n", i, X_serial_r[i], X_serial_i[i]);
		printf("%d:\t%f\t%f\n", i, X_r[i], X_i[i]);
	}
	*/
	
	
	
    // free allocated memory for later use
    free(x_r);
    free(x_i);
    free(X_serial_r);
    free(X_serial_i);
    free(X_r);
    free(X_i);

    return 0;
}



//-----------------------------------------------------------------------------
/**
 * @brief Executes FFT on the GPU using either a simple or efficient kernel.
 *
 * This function allocates memory on the GPU, transfers input data from the host to the device,
 * executes the selected FFT kernel, and transfers the results back to the host. It also measures
 * the time taken by the GPU kernel execution.
 *
 * @param x_r Pointer to the real part of the input signal on the host.
 * @param x_i Pointer to the imaginary part of the input signal on the host.
 * @param X_r Pointer to the real part of the output signal on the host.
 * @param X_i Pointer to the imaginary part of the output signal on the host.
 * @param N The number of elements in the input signal.
 * @param M The number of stages in the FFT.
 * @param SIMPLE Flag to select between simple (1) and efficient (0) FFT kernel.
 * @param gpu_kernel_time Pointer to a double variable to store the elapsed time of the GPU kernel execution.
 */
void gpuKernels(float* x_r, float* x_i, float* X_r, float* X_i, unsigned int N, unsigned int M, unsigned int SIMPLE, double* gpu_kernel_time) {
    float* x_r_d;
    float* x_i_d;
    float* X_r_d;
    float* X_i_d;

    HANDLE_ERROR(cudaMalloc((void**)&x_r_d, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&x_i_d, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&X_r_d, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&X_i_d, N * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(x_r_d, x_r, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(x_i_d, x_i, N * sizeof(float), cudaMemcpyHostToDevice));

	GpuTimer timer;
    timer.Start();
	if (SIMPLE==1) gpuKernel_simple   (x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	else           gpuKernel_efficient(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	timer.Stop();
	*gpu_kernel_time = timer.Elapsed();
	
    HANDLE_ERROR(cudaMemcpy(X_r, X_r_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(X_i, X_i_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(x_r_d));
    HANDLE_ERROR(cudaFree(x_i_d));
    HANDLE_ERROR(cudaFree(X_r_d));
    HANDLE_ERROR(cudaFree(X_i_d));
}



//-----------------------------------------------------------------------------
/**
 * @brief Perform the Fast Fourier Transform (FFT) on the input arrays.
 *
 * This function recursively computes the FFT of the input arrays X_serial_r and X_serial_i,
 * which represent the real and imaginary parts of the input signal, respectively.
 *
 * @param X_serial_r Pointer to the array containing the real part of the input signal.
 * @param X_serial_i Pointer to the array containing the imaginary part of the input signal.
 * @param n The size of the input arrays. Must be a power of 2.
 * @param tmp_r Pointer to a temporary array for storing intermediate real values.
 * @param tmp_i Pointer to a temporary array for storing intermediate imaginary values.
 *
 * The function performs the FFT by recursively dividing the input arrays into even and odd
 * indexed elements, computing the FFT on these subarrays, and then combining the results.
 * The results are stored back in the input arrays X_serial_r and X_serial_i.
 */
void cpuKernel(float* X_serial_r, float* X_serial_i, int n, float* tmp_r, float* tmp_i) {
	if(n > 1) {	// otherwise, do nothing and return
		int k, m;
		float z_r, z_i, w_r, w_i;
		float *vo_r, *vo_i, *ve_r, *ve_i;
		ve_r = tmp_r; ve_i = tmp_i;
		vo_r = tmp_r + n/2; vo_i = tmp_i + n/2;
		
		for(k=0; k<n/2; k++) {
			ve_r[k] = X_serial_r[2*k]; ve_i[k] = X_serial_i[2*k];
			vo_r[k] = X_serial_r[2*k+1]; vo_i[k] = X_serial_i[2*k+1];
		}
		cpuKernel(ve_r, ve_i, n/2, X_serial_r, X_serial_i);	// FFT on even-indexed elements of v[]
		cpuKernel(vo_r, vo_i, n/2, X_serial_r, X_serial_i);	// FFT on odd-indexed elements of v[]
		
		for(m=0; m<n/2; m++) {
			w_r =  cos((2*PI*m)/n);
			w_i = -sin((2*PI*m)/n);
			z_r = w_r*vo_r[m] - w_i*vo_i[m];	// Re(w*vo[m])
			z_i = w_r*vo_i[m] + w_i*vo_r[m];	// Im(w*vo[m])
			X_serial_r[  m  ] = ve_r[m] + z_r;
			X_serial_i[  m  ] = ve_i[m] + z_i;
			X_serial_r[m+n/2] = ve_r[m] - z_r;
			X_serial_i[m+n/2] = ve_i[m] - z_i;
		}
	}
	return;
}


//-----------------------------------------------------------------------------
/**
 * @brief Parses and validates command line arguments to set FFT parameters.
 *
 * This function takes command line arguments and extracts the values for SIMPLE and M.
 * It also calculates the value of N as 2 raised to the power of M.
 * If the arguments are invalid, it prints an error message and exits the program.
 *
 * @param argc The number of command line arguments.
 * @param argv The array of command line arguments.
 * @param N Reference to an unsigned int where the calculated value of N will be stored.
 * @param M Reference to an unsigned int where the value of M will be stored.
 * @param SIMPLE Reference to an unsigned int where the value of SIMPLE will be stored.
 *
 * @note The program expects exactly 3 arguments:
 *       - argv[1]: SIMPLE (must be 0 or 1)
 *       - argv[2]: M (must be between 0 and 25)
 *       If the arguments do not meet these criteria, the program will print an error message and exit.
 */
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& M, unsigned int& SIMPLE)
{
    if (
	argc != 3 || 
	atoi(argv[1]) < 0 || atoi(argv[1]) > 1 || 
	atoi(argv[2]) < 0 || atoi(argv[2]) > 25 
	) {
        printf("<< Error >>\n");
        printf("Enter the following command:\n");
        printf("\t./a.out  SIMPLE  M\n");
        printf("\t\tSIMPLE must be 0 or 1\n");
        printf("\t\tM must be between 0 and 25\n");
		exit(-1);
    }
	SIMPLE = atoi(argv[1]);
	M = atoi(argv[2]);
    N = (1 << M);
}


//-----------------------------------------------------------------------------
/**
 * @brief Fills an array with random float values.
 *
 * This function populates the provided array with random float values
 * in the range of -8 to 8.
 *
 * @param data Pointer to the array to be filled.
 * @param size The number of elements in the array.
 */
void fill(float* data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = (float)(rand() % 17 - 8);
}


/**
 * @brief Calculates the Mean Squared Error (MSE) between two sets of complex data.
 *
 * This function computes the MSE between two complex data sets represented by their
 * real and imaginary parts. The MSE is calculated as the sum of the squared differences
 * between corresponding real and imaginary parts of the two data sets.
 *
 * @param data1_r Pointer to the array containing the real parts of the first data set.
 * @param data1_i Pointer to the array containing the imaginary parts of the first data set.
 * @param data2_r Pointer to the array containing the real parts of the second data set.
 * @param data2_i Pointer to the array containing the imaginary parts of the second data set.
 * @param size The number of elements in each data set.
 * @return The calculated Mean Squared Error (MSE) as a double.
 */
double calc_mse(float* data1_r, float* data1_i, float* data2_r, float* data2_i, int size) {
    double mse = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        double e_r = data1_r[i] - data2_r[i];
        double e_i = data1_i[i] - data2_i[i];
        double e = e_r * e_r + e_i * e_i;
        mse += e;
    }
    return mse;
}

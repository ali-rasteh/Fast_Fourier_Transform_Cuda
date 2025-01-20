//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"
#include <stdio.h>
#include <stdlib.h>


#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z


// you may define other parameters here!

#define max_threads			512
#define max_threads_log2	9




// you may define other macros here!
// you may define other functions here!




/**
 * @brief Calculates the bit-reversed index for a given index and number of digits.
 *
 * This function computes the bit-reversed index of the input index. The bit-reversed
 * index is calculated by reversing the order of the bits in the input index up to
 * the specified number of digits.
 *
 * @param index The input index to be bit-reversed.
 * @param digits The number of bits to consider for the bit-reversal.
 * @return The bit-reversed index.
 */
__device__ int shuffled_index_cal (int index , int digits){
	int i;
	int out_index = 0;
	for (i=0 ; i<digits ; i++){
		out_index += (((index>>i)&(0x1))<<(digits-1-i)) & (1<<(digits-1-i));
	}
	return out_index;
}




/**
 * @brief Computes 2 raised to the power of the given integer.
 *
 * This function calculates 2 raised to the power of the input integer `in`.
 * It uses a simple loop to multiply 2 by itself `in` times.
 *
 * @param in The exponent to which 2 is raised.
 * @return The result of 2 raised to the power of `in`.
 */
__device__ int power2(int in){
	int i;
	int in_power2 = 1;
	for (i=0 ; i<in ; i++){
		in_power2*=2;
	}
	return in_power2;
}



/**
 * @brief Computes the base-2 logarithm of an integer.
 *
 * This function calculates the base-2 logarithm of the input integer by
 * repeatedly dividing the input by 2 until it reaches 1, counting the number
 * of divisions performed.
 *
 * @param in The input integer for which the base-2 logarithm is to be computed.
 * @return The base-2 logarithm of the input integer.
 */
__device__ int log2(int in){
	int in1 = in;
	int in_log2 = 0;
	while (in1!=1){
		in1/=2;
		in_log2+=1;
	}
	return in_log2;
}




/**
 * @brief Performs the Fast Fourier Transform (FFT) calculation on the input arrays.
 *
 * This function computes the FFT of the input arrays `in_r` and `in_i` using the Cooley-Tukey algorithm.
 * The input arrays are expected to be in bit-reversed order.
 *
 * @param in_r Pointer to the real part of the input array.
 * @param in_i Pointer to the imaginary part of the input array.
 * @param N The size of the input arrays, which must be a power of 2.
 *
 * @note This function is intended to be called from within a CUDA kernel.
 *       It uses shared memory and synchronization primitives to perform the FFT in parallel.
 *
 * @internal
 * The function performs the following steps:
 * 1. Bit-reversal permutation of the input arrays.
 * 2. Iterative computation of the FFT using the Cooley-Tukey algorithm.
 * 3. Synchronization of threads at each step to ensure correct computation.
 *
 * @warning The input size `N` must be a power of 2.
 * @warning This function assumes that the number of threads is equal to `N/2`.
 */
__device__ void FFT_calc(float* in_r, float* in_i, const unsigned int N){
	
	int k , shuffled_index;
	int thread;
	int index;
	int step;
	int step_power2;
	float tmp1_r, tmp1_i, tmp2_r, tmp2_i;
	float z_r, z_i, w_r, w_i;
	float vo_r, vo_i, ve_r, ve_i;
	
	int N1 = N;
	int N_log2 = log2(N1);
	
	thread = tx;
	
	shuffled_index = shuffled_index_cal(thread , N_log2);
	
	tmp1_r = in_r[shuffled_index];
	tmp1_i = in_i[shuffled_index];
	tmp2_r = in_r[shuffled_index+1];
	tmp2_i = in_i[shuffled_index+1];
	
	__syncthreads();
	
	in_r[thread] = tmp1_r;
	in_i[thread] = tmp1_i;
	in_r[thread+(N/2)] = tmp2_r;
	in_i[thread+(N/2)] = tmp2_i;
	
	__syncthreads();
	
	
	step_power2 = 1;
	
	for(step=0 ; step<N_log2 ; step++){
		index = ((int)(thread/step_power2)) * (2*step_power2) + thread%step_power2;
		
		ve_r = in_r[index]; ve_i = in_i[index];
		vo_r = in_r[index + step_power2]; vo_i = in_i[index + step_power2];
		
		k = thread%step_power2;
		w_r =  cos((2*PI*k)/(2*step_power2));
		w_i = -sin((2*PI*k)/(2*step_power2));
		z_r = w_r*vo_r - w_i*vo_i;	// Re(w*vo)
		z_i = w_r*vo_i + w_i*vo_r;	// Im(w*vo)
		in_r[index] = ve_r + z_r;
		in_i[index] = ve_i + z_i;
		in_r[index + step_power2] = ve_r - z_r;
		in_i[index + step_power2] = ve_i - z_i;
		__syncthreads();
		step_power2*=2;
	}
	
}




//-----------------------------------------------------------------------------
/**
 * @brief CUDA kernel function to perform a step in the Fast Fourier Transform (FFT) computation.
 *
 * @param x_r_d Pointer to the real part of the input data array in device memory.
 * @param x_i_d Pointer to the imaginary part of the input data array in device memory.
 * @param X_r_d Pointer to the real part of the output data array in device memory.
 * @param X_i_d Pointer to the imaginary part of the output data array in device memory.
 * @param N The total number of elements in the input data array.
 * @param M The current stage of the FFT computation.
 * @param step The current step in the FFT computation.
 *
 * @details This kernel function performs a step in the FFT computation. Depending on the value of the step parameter,
 *          it either shuffles the input data or performs the butterfly computation. The function uses shared memory
 *          to optimize memory access patterns and improve performance.
 *
 * @note The function assumes that the input data is already in bit-reversed order.
 */
__global__ void kernelFunc_simple(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M , unsigned int step) 
{
	int k , shuffled_index;
	int thread;
	int index;
	int step_power2;
	
	float z_r, z_i, w_r, w_i;
	float vo_r, vo_i, ve_r, ve_i;
	
	//cuPrintf("kernelFunc_simple is getting started\n");
	
	
	if(M<11){
		thread = tx;
	}
	else{
		thread = bx * 512 + tx;
	}
	
	
	if (step == 50){
		
		shuffled_index = shuffled_index_cal(thread , M);
		
		X_r_d[thread] = x_r_d[shuffled_index];
		X_i_d[thread] = x_i_d[shuffled_index];
		
		X_r_d[thread+(N/2)] = x_r_d[shuffled_index+1];
		X_i_d[thread+(N/2)] = x_i_d[shuffled_index+1];
	}
	
	
	else if (step != 50){

		step_power2 = power2(step);
		
		index = ((int)(thread/step_power2)) * (2*step_power2) + thread%step_power2;
		
		ve_r = X_r_d[index]; ve_i = X_i_d[index];
		vo_r = X_r_d[index + step_power2]; vo_i = X_i_d[index + step_power2];
		
		k = thread%step_power2;
		w_r =  cos((2*PI*k)/(2*step_power2));
		w_i = -sin((2*PI*k)/(2*step_power2));
		z_r = w_r*vo_r - w_i*vo_i;	// Re(w*vo)
		z_i = w_r*vo_i + w_i*vo_r;	// Im(w*vo)
		X_r_d[index] = ve_r + z_r;
		X_i_d[index] = ve_i + z_i;
		X_r_d[index + step_power2] = ve_r - z_r;
		X_i_d[index + step_power2] = ve_i - z_i;
	}
	
}




//-----------------------------------------------------------------------------
/**
 * @brief CUDA kernel function to perform a simple FFT computation.
 *
 * This kernel function performs a Fast Fourier Transform (FFT) on the input data arrays.
 * It uses shared memory to optimize memory access and performs the computation in parallel
 * using CUDA threads.
 *
 * @param x_r_d Pointer to the real part of the input data array.
 * @param x_i_d Pointer to the imaginary part of the input data array.
 * @param X_r_d Pointer to the real part of the output data array.
 * @param X_i_d Pointer to the imaginary part of the output data array.
 * @param N The total number of points in the FFT.
 * @param M The number of stages in the FFT.
 *
 * @note This function assumes that the input data arrays are already allocated and initialized.
 *       The function uses shared memory to store intermediate results and synchronizes threads
 *       using __syncthreads().
 */
__global__ void kernelFunc_simple1(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{

	int thread, point_ID;
	int N1;
	int k;
	int index;
	int step, step_power2;
	
	float z_r, z_i, w_r, w_i;
	float vo_r, vo_i, ve_r, ve_i;
	
	
	__shared__ float X_r_d_shared[1024];
	__shared__ float X_i_d_shared[1024];
	
	
	if(M<11){
		thread = tx;
	}
	else{
		thread = bx * 512 + tx;
	}
	
	
	N1 = max_threads * 2;
	point_ID = bx*N1 + tx;
	thread = tx;

	
	X_r_d_shared[thread] = X_r_d[point_ID];
	X_i_d_shared[thread] = X_i_d[point_ID];
	
	X_r_d_shared[thread + N1/2] = X_r_d[point_ID + N1/2];
	X_i_d_shared[thread + N1/2] = X_i_d[point_ID + N1/2];
	__syncthreads();
	
	
	step_power2 = 1;
	
	for(step=0 ; step<10 ; step++){

		index = ((int)(thread/step_power2)) * (2*step_power2) + thread%step_power2;
		
		ve_r = X_r_d_shared[index]; ve_i = X_i_d_shared[index];
		vo_r = X_r_d_shared[index + step_power2]; vo_i = X_i_d_shared[index + step_power2];
		
		k = thread%step_power2;
		w_r =  cos((2*PI*k)/(2*step_power2));
		w_i = -sin((2*PI*k)/(2*step_power2));
		z_r = w_r*vo_r - w_i*vo_i;	// Re(w*vo)
		z_i = w_r*vo_i + w_i*vo_r;	// Im(w*vo)
		X_r_d_shared[index] = ve_r + z_r;
		X_i_d_shared[index] = ve_i + z_i;
		X_r_d_shared[index + step_power2] = ve_r - z_r;
		X_i_d_shared[index + step_power2] = ve_i - z_i;
		__syncthreads();
		step_power2*=2;
	}
	
	
	X_r_d[point_ID] = X_r_d_shared[thread];
	X_i_d[point_ID] = X_i_d_shared[thread];
	
	X_r_d[point_ID+(N1/2)] = X_r_d_shared[thread+(N1/2)];
	X_i_d[point_ID+(N1/2)] = X_i_d_shared[thread+(N1/2)];
	
}





//-----------------------------------------------------------------------------
/**
 * @brief Efficient kernel function for performing Fast Fourier Transform (FFT) on complex numbers.
 *
 * This kernel function uses shared memory to perform FFT calculations on complex numbers represented
 * by their real and imaginary parts. The input arrays are divided into two halves and loaded into shared
 * memory for efficient computation.
 *
 * @param x_r_d Pointer to the array containing the real parts of the input complex numbers.
 * @param x_i_d Pointer to the array containing the imaginary parts of the input complex numbers.
 * @param X_r_d Pointer to the array where the real parts of the output complex numbers will be stored.
 * @param X_i_d Pointer to the array where the imaginary parts of the output complex numbers will be stored.
 * @param N The total number of complex numbers in the input arrays.
 * @param M The number of threads per block (not used in the current implementation).
 */
__global__ void kernelFunc_efficient(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	int thread;
	
	
	__shared__ float X_r_d_shared[1024];
	__shared__ float X_i_d_shared[1024];
	
	
	thread = tx;

	X_r_d_shared[thread] = x_r_d[thread];
	X_i_d_shared[thread] = x_i_d[thread];
	
	X_r_d_shared[thread + N/2] = x_r_d[thread + N/2];
	X_i_d_shared[thread + N/2] = x_i_d[thread + N/2];
	
	__syncthreads();
	
	
	FFT_calc(X_r_d_shared , X_i_d_shared , N);
	
	X_r_d[thread] = X_r_d_shared[thread];
	X_i_d[thread] = X_i_d_shared[thread];
	
	X_r_d[thread+(N/2)] = X_r_d_shared[thread+(N/2)];
	X_i_d[thread+(N/2)] = X_i_d_shared[thread+(N/2)];

}





//-----------------------------------------------------------------------------
/**
 * @brief Efficient kernel function for computing the Fast Fourier Transform (FFT) on CUDA.
 *
 * This kernel function performs the FFT on input arrays of real and imaginary parts.
 * It utilizes shared memory for efficient computation and performs the FFT in parallel.
 *
 * @param x_r_d Pointer to the input array of real parts.
 * @param x_i_d Pointer to the input array of imaginary parts.
 * @param X_r_d Pointer to the output array of real parts.
 * @param X_i_d Pointer to the output array of imaginary parts.
 * @param N Total number of points in the FFT.
 * @param M Number of stages in the FFT.
 *
 * @note This kernel assumes that the input arrays are already allocated and initialized on the device.
 *       The shared memory size is fixed at 1024 elements for both real and imaginary parts.
 *       The function `FFT_calc` is assumed to be defined elsewhere and performs the FFT calculation.
 *
 * @note The kernel uses a block size of `max_threads` and processes `N1` points per block.
 *       The input arrays are divided into smaller chunks and processed in parallel.
 *       The results are stored back into the global memory after computation.
 */
__global__ void kernelFunc_efficient1(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	
	int thread, point_ID;
	int N1, N2;
	
	float z_r, z_i, w_r, w_i;
	
	__shared__ float X_r_d_shared[1024];
	__shared__ float X_i_d_shared[1024];
	
	
	N1 = max_threads * 2;
	N2 = N/N1;
	
	
	
	point_ID = tx*N1 + bx;
	thread = tx;
	
	X_r_d_shared[thread] = x_r_d[point_ID];
	X_i_d_shared[thread] = x_i_d[point_ID];
	
	X_r_d_shared[thread + N2/2] = x_r_d[point_ID + (N2/2)*N1];
	X_i_d_shared[thread + N2/2] = x_i_d[point_ID + (N2/2)*N1];
	
	__syncthreads();
	
	
	FFT_calc(X_r_d_shared , X_i_d_shared , N2);
	
	
	
	w_r =  cos((2*PI*(thread)*(bx))/N);
	w_i = -sin((2*PI*(thread)*(bx))/N);
	z_r = w_r*X_r_d_shared[thread] - w_i*X_i_d_shared[thread];	// Re(w*X_d_shared)
	z_i = w_r*X_i_d_shared[thread] + w_i*X_r_d_shared[thread];	// Im(w*X_d_shared)
	X_r_d_shared[thread] = z_r;
	X_i_d_shared[thread] = z_i;
	
	w_r =  cos((2*PI*(thread+N2/2)*(bx))/N);
	w_i = -sin((2*PI*(thread+N2/2)*(bx))/N);
	z_r = w_r*X_r_d_shared[thread+N2/2] - w_i*X_i_d_shared[thread+N2/2];	// Re(w*X_d_shared)
	z_i = w_r*X_i_d_shared[thread+N2/2] + w_i*X_r_d_shared[thread+N2/2];	// Im(w*X_d_shared)
	X_r_d_shared[thread+N2/2] = z_r;
	X_i_d_shared[thread+N2/2] = z_i;
	
	
	X_r_d[point_ID] = X_r_d_shared[thread];
	X_i_d[point_ID] = X_i_d_shared[thread];
	
	X_r_d[point_ID+(N2/2)*N1] = X_r_d_shared[thread+(N2/2)];
	X_i_d[point_ID+(N2/2)*N1] = X_i_d_shared[thread+(N2/2)];
	
	
}





//-----------------------------------------------------------------------------
/**
 * @brief Efficient kernel function for performing FFT on complex data.
 *
 * This kernel function performs an efficient Fast Fourier Transform (FFT) on 
 * complex data using shared memory for intermediate storage and transposition 
 * of data for coalesced memory access.
 *
 * @param x_r_d Pointer to the real part of the input data array in device memory.
 * @param x_i_d Pointer to the imaginary part of the input data array in device memory.
 * @param X_r_d Pointer to the real part of the output data array in device memory.
 * @param X_i_d Pointer to the imaginary part of the output data array in device memory.
 * @param N Total number of points in the FFT.
 * @param M Number of points in each FFT segment.
 *
 * @note This kernel assumes that the number of threads per block (max_threads) 
 *       is set appropriately and that N is a multiple of max_threads * 2.
 */
__global__ void kernelFunc_efficient2(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	int thread, point_ID, point_ID_transpose;
	int N1, N2;
	
	
	__shared__ float X_r_d_shared[max_threads * 2];
	__shared__ float X_i_d_shared[max_threads * 2];
	
	N2 = max_threads * 2;
	N1 = N/N2;
	
	point_ID = bx*N2 + tx;
	thread = tx;
	
	
	
	X_r_d_shared[thread] = X_r_d[point_ID];
	X_i_d_shared[thread] = X_i_d[point_ID];
	
	X_r_d_shared[thread + N2/2] = X_r_d[point_ID + N2/2];
	X_i_d_shared[thread + N2/2] = X_i_d[point_ID + N2/2];
	
	__syncthreads();
	
	FFT_calc(X_r_d_shared , X_i_d_shared , N2);
	
	
	point_ID_transpose = tx * N1 + bx;
	x_r_d[point_ID_transpose] = X_r_d_shared[thread];			//X_r_d[point_ID]
	x_i_d[point_ID_transpose] = X_i_d_shared[thread];			//X_i_d[point_ID]
	
	x_r_d[(tx+N2/2)*N1 + bx] = X_r_d_shared[thread+(N2/2)];		//X_r_d[point_ID+(N2/2)]
	x_i_d[(tx+N2/2)*N1 + bx] = X_i_d_shared[thread+(N2/2)];		//X_i_d[point_ID+(N2/2)]
	
}





//-----------------------------------------------------------------------------
/**
 * @brief CUDA kernel function to efficiently copy input arrays to output arrays.
 *
 * This kernel function copies elements from input arrays `x_r_d` and `x_i_d` to 
 * output arrays `X_r_d` and `X_i_d`. The copying is done in a manner that 
 * leverages the maximum number of threads available.
 *
 * @param x_r_d Pointer to the real part of the input array.
 * @param x_i_d Pointer to the imaginary part of the input array.
 * @param X_r_d Pointer to the real part of the output array.
 * @param X_i_d Pointer to the imaginary part of the output array.
 * @param N Total number of elements in the input arrays.
 * @param M Number of elements to be processed by each thread block.
 */
__global__ void kernelFunc_efficient3(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	int point_ID;
	int N2;
	
	N2 = max_threads * 2;
	
	point_ID = bx*N2 + tx;
	
	X_r_d[point_ID] = x_r_d[point_ID];
	X_i_d[point_ID] = x_i_d[point_ID];
	
	X_r_d[point_ID + N2/2] = x_r_d[point_ID + N2/2];
	X_i_d[point_ID + N2/2] = x_i_d[point_ID + N2/2];
}







//-----------------------------------------------------------------------------
/**
 * @brief CUDA kernel function to efficiently copy input arrays to output arrays.
 *
 * This kernel function copies elements from input arrays `x_r_d` and `x_i_d` to 
 * output arrays `X_r_d` and `X_i_d`. The copying is done in a manner that 
 * leverages the maximum number of threads available.
 *
 * @param x_r_d Pointer to the real part of the input array.
 * @param x_i_d Pointer to the imaginary part of the input array.
 * @param X_r_d Pointer to the real part of the output array.
 * @param X_i_d Pointer to the imaginary part of the output array.
 * @param N Total number of elements in the input arrays.
 * @param M Number of elements to be processed by each thread block.
 */
__global__ void kernelFunc_efficient4(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	int thread, point_ID_1, point_ID_2;
	int N1, N2, N3;
	
	float z_r, z_i, w_r, w_i;
	
	__shared__ float X_r_d_shared[1024];
	__shared__ float X_i_d_shared[1024];
	
	
	N1 = max_threads * 2;
	N2 = N/N1;
	N3 = N2/N1;
	
	
	point_ID_1 = (tx*N1+(bx%N1))*N1 + bx/N1;
	point_ID_2 = ((tx+N3/2)*N1+(bx%N1))*N1 + bx/N1;
	
	thread = tx;
	
	X_r_d_shared[thread] = x_r_d[point_ID_1];
	X_i_d_shared[thread] = x_i_d[point_ID_1];
	
	X_r_d_shared[thread + N3/2] = x_r_d[point_ID_2];
	X_i_d_shared[thread + N3/2] = x_i_d[point_ID_2];
	
	__syncthreads();
	
	
	FFT_calc(X_r_d_shared , X_i_d_shared , N3);
	
	
	
	w_r =  cos((2*PI*(thread)*(bx%N1))/N2);
	w_i = -sin((2*PI*(thread)*(bx%N1))/N2);
	z_r = w_r*X_r_d_shared[thread] - w_i*X_i_d_shared[thread];	// Re(w*X_d_shared)
	z_i = w_r*X_i_d_shared[thread] + w_i*X_r_d_shared[thread];	// Im(w*X_d_shared)
	X_r_d_shared[thread] = z_r;
	X_i_d_shared[thread] = z_i;
	
	w_r =  cos((2*PI*(thread+N3/2)*(bx%N1))/N2);
	w_i = -sin((2*PI*(thread+N3/2)*(bx%N1))/N2);
	z_r = w_r*X_r_d_shared[thread+N3/2] - w_i*X_i_d_shared[thread+N3/2];	// Re(w*X_d_shared)
	z_i = w_r*X_i_d_shared[thread+N3/2] + w_i*X_r_d_shared[thread+N3/2];	// Im(w*X_d_shared)
	X_r_d_shared[thread+N3/2] = z_r;
	X_i_d_shared[thread+N3/2] = z_i;
	
	__syncthreads();
	
	X_r_d[point_ID_1] = X_r_d_shared[thread];
	X_i_d[point_ID_1] = X_i_d_shared[thread];
	
	X_r_d[point_ID_2] = X_r_d_shared[thread+(N3/2)];
	X_i_d[point_ID_2] = X_i_d_shared[thread+(N3/2)];
	
}




//-----------------------------------------------------------------------------
/**
 * @brief CUDA kernel function to efficiently copy input arrays to output arrays.
 *
 * This kernel function copies elements from input arrays `x_r_d` and `x_i_d` to 
 * output arrays `X_r_d` and `X_i_d`. The copying is done in a manner that 
 * leverages the maximum number of threads available.
 *
 * @param x_r_d Pointer to the real part of the input array.
 * @param x_i_d Pointer to the imaginary part of the input array.
 * @param X_r_d Pointer to the real part of the output array.
 * @param X_i_d Pointer to the imaginary part of the output array.
 * @param N Total number of elements in the input arrays.
 * @param M Number of elements to be processed by each thread block.
 */
__global__ void kernelFunc_efficient5(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	int thread, point_ID_1, point_ID_2, point_ID_transpose_1, point_ID_transpose_2;
	int N1, N2, N3;
	
	
	__shared__ float X_r_d_shared[max_threads * 2];
	__shared__ float X_i_d_shared[max_threads * 2];
	
	
	N1 = max_threads * 2;
	N2 = N/N1;
	N3 = N2/N1;
	
	
	point_ID_1 = ((bx%N3)*N1+tx)*N1 + bx/N3;
	point_ID_2 = ((bx%N3)*N1+(tx+N1/2))*N1 + bx/N3;
	
	thread = tx;
	
	
	
	X_r_d_shared[thread] = X_r_d[point_ID_1];
	X_i_d_shared[thread] = X_i_d[point_ID_1];
	
	X_r_d_shared[thread + N1/2] = X_r_d[point_ID_2];
	X_i_d_shared[thread + N1/2] = X_i_d[point_ID_2];
	
	__syncthreads();
	
	FFT_calc(X_r_d_shared , X_i_d_shared , N1);
	
	
	point_ID_transpose_1 = (tx*N3+(bx%N3))*N1 + bx/N3;
	point_ID_transpose_2 = ((tx+N1/2)*N3+(bx%N3))*N1 + bx/N3;
	
	x_r_d[point_ID_transpose_1] = X_r_d_shared[thread];			//X_r_d[point_ID]
	x_i_d[point_ID_transpose_1] = X_i_d_shared[thread];			//X_i_d[point_ID]
	
	x_r_d[point_ID_transpose_2] = X_r_d_shared[thread+(N1/2)];		//X_r_d[point_ID+(N2/2)]
	x_i_d[point_ID_transpose_2] = X_i_d_shared[thread+(N1/2)];		//X_i_d[point_ID+(N2/2)]
	
}




//-----------------------------------------------------------------------------
/**
 * @brief CUDA kernel function to efficiently copy input arrays to output arrays.
 *
 * This kernel function copies elements from input arrays `x_r_d` and `x_i_d` to 
 * output arrays `X_r_d` and `X_i_d`. The copying is done in a manner that 
 * leverages the maximum number of threads available.
 *
 * @param x_r_d Pointer to the real part of the input array.
 * @param x_i_d Pointer to the imaginary part of the input array.
 * @param X_r_d Pointer to the real part of the output array.
 * @param X_i_d Pointer to the imaginary part of the output array.
 * @param N Total number of elements in the input arrays.
 * @param M Number of elements to be processed by each thread block.
 */
__global__ void kernelFunc_efficient6(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M) 
{
	int thread, point_ID, point_ID_transpose;
	int N1, N2;

	float z_r, z_i, w_r, w_i;
	
	__shared__ float X_r_d_shared[max_threads * 2];
	__shared__ float X_i_d_shared[max_threads * 2];
	
	N1 = max_threads * 2;
	N2 = N/N1;
	
	point_ID = bx*N1 + tx;
	thread = tx;
	
	
	X_r_d_shared[thread] = x_r_d[point_ID];
	X_i_d_shared[thread] = x_i_d[point_ID];
	
	X_r_d_shared[thread + N1/2] = x_r_d[point_ID + N1/2];
	X_i_d_shared[thread + N1/2] = x_i_d[point_ID + N1/2];
	
	__syncthreads();
	
	w_r =  cos((2*PI*(bx)*(thread))/N);
	w_i = -sin((2*PI*(bx)*(thread))/N);
	z_r = w_r*X_r_d_shared[thread] - w_i*X_i_d_shared[thread];	// Re(w*X_d_shared)
	z_i = w_r*X_i_d_shared[thread] + w_i*X_r_d_shared[thread];	// Im(w*X_d_shared)
	X_r_d_shared[thread] = z_r;
	X_i_d_shared[thread] = z_i;
	
	w_r =  cos((2*PI*(bx)*(thread+N1/2))/N);
	w_i = -sin((2*PI*(bx)*(thread+N1/2))/N);
	z_r = w_r*X_r_d_shared[thread+N1/2] - w_i*X_i_d_shared[thread+N1/2];	// Re(w*X_d_shared)
	z_i = w_r*X_i_d_shared[thread+N1/2] + w_i*X_r_d_shared[thread+N1/2];	// Im(w*X_d_shared)
	X_r_d_shared[thread+N1/2] = z_r;
	X_i_d_shared[thread+N1/2] = z_i;
	
	__syncthreads();
	

	FFT_calc(X_r_d_shared , X_i_d_shared , N1);
	
	
	point_ID_transpose = tx * N2 + bx;
	X_r_d[point_ID_transpose] = X_r_d_shared[thread];			//X_r_d[point_ID]
	X_i_d[point_ID_transpose] = X_i_d_shared[thread];			//X_i_d[point_ID]
	
	X_r_d[(tx+N1/2)*N2 + bx] = X_r_d_shared[thread+(N1/2)];		//X_r_d[point_ID+(N2/2)]
	X_i_d[(tx+N1/2)*N2 + bx] = X_i_d_shared[thread+(N1/2)];		//X_i_d[point_ID+(N2/2)]
	
}






//-----------------------------------------------------------------------------
/**
 * @brief GPU kernel function for performing a simple FFT computation.
 *
 * This function executes a simple FFT computation on the GPU. Both input and output arrays are 
 * expected to be already allocated on the GPU. The function does not perform any memory 
 * allocation or data transfer between the host and the device.
 *
 * @param x_r_d Pointer to the real part of the input array on the GPU.
 * @param x_i_d Pointer to the imaginary part of the input array on the GPU.
 * @param X_r_d Pointer to the real part of the output array on the GPU.
 * @param X_i_d Pointer to the imaginary part of the output array on the GPU.
 * @param N The size of the input array.
 * @param M The number of steps in the FFT computation.
 */
void gpuKernel_simple(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{
	// In this function, both inputs and outputs are on GPU.
	// No need for cudaMalloc, cudaMemcpy or cudaFree.
	
	int step;
	int gridx , gridy , blockx , blocky;
	
	if (M<11){
		gridx = gridy = 1;
		blockx = N/2;
		blocky = 1;
	}
	else{
		gridx = N/(2*max_threads);
		gridy = 1;
		blockx = max_threads;
		blocky = 1;	
	}
	
	//cudaPrintfInit();
    //cudaPrintfDisplay(stdout, true);
    
	
	dim3 dimGrid(gridx,gridy);
	dim3 dimBlock(blockx,blocky);
	
	kernelFunc_simple <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M ,50);	
	
	for(step=0 ; step<M ; step++){
		kernelFunc_simple <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M ,step);
	}
	
	//cudaPrintfEnd();
}





//-----------------------------------------------------------------------------
/**
 * @brief Perform an efficient GPU-based Fast Fourier Transform (FFT).
 *
 * This function executes a series of CUDA kernels to perform an FFT on the input data.
 * The input and output data are both located on the GPU, so there is no need for 
 * cudaMalloc, cudaMemcpy, or cudaFree operations within this function.
 *
 * @param x_r_d Pointer to the real part of the input data on the GPU.
 * @param x_i_d Pointer to the imaginary part of the input data on the GPU.
 * @param X_r_d Pointer to the real part of the output data on the GPU.
 * @param X_i_d Pointer to the imaginary part of the output data on the GPU.
 * @param N The total number of data points.
 * @param M The number of stages in the FFT.
 *
 * The function contains several commented-out sections of code that demonstrate different
 * configurations for executing the FFT, including:
 * - Quasi simple code 1: A basic implementation for small values of M.
 * - Quasi simple code 2: Another basic implementation with a different configuration.
 * - 1D FFT: A configuration that works for M < 11.
 * - 2D FFT: A configuration that works for M < 21.
 * - 3D FFT: The current implementation, which works for M < 31.
 *
 * The current implementation uses three kernel functions (kernelFunc_efficient4, 
 * kernelFunc_efficient5, and kernelFunc_efficient6) to perform the FFT in three stages.
 * Each stage uses different grid and block dimensions to optimize performance.
 */
void gpuKernel_efficient(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M)
{
	// In this function, both inputs and outputs are on GPU.
	// No need for cudaMalloc, cudaMemcpy or cudaFree.

	int N1 = max_threads * 2;
	int N2 = N/N1;
	int N3 = N2/N1;
	
	int step;
	int gridx , gridy , blockx , blocky;
	
//----------------------------------------------------------------------------- quasi simple code 1	
	/*
	if (M<11){
		gridx = gridy = 1;
		blockx = N/2;
		blocky = 1;
	}
	else{
		gridx = N/(2*max_threads);
		gridy = 1;
		blockx = max_threads;
		blocky = 1;	
	}
	
	dim3 dimGrid(gridx,gridy);
	dim3 dimBlock(blockx,blocky);
	
	kernelFunc_simple <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M ,50);	
	
	for(step=0 ; step<M ; step++){
		kernelFunc_simple <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M ,step);
	}
	*/
//----------------------------------------------------------------------------- quasi simple code 2
	/*
	gridx = N/(2*max_threads);
	gridy = 1;
	blockx = max_threads;
	blocky = 1;	
	
	
	dim3 dimGrid(gridx,gridy);
	dim3 dimBlock(blockx,blocky);
	
	
	kernelFunc_simple <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M ,50);
	
	kernelFunc_simple1 <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	
	for(step=10 ; step<M ; step++){
		kernelFunc_simple <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M ,step);
	}
	*/
//----------------------------------------------------------------------------- 1D FFT (working for M<11)
	/*
	dim3 dimGrid(1,1);
	dim3 dimBlock(N/2,1);
	kernelFunc_efficient <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	*/

//----------------------------------------------------------------------------- 2D FFT	(working for M<21)
	/*
	dim3 dimGrid1(N1,1);
	dim3 dimBlock1(N2/2,1);
	kernelFunc_efficient1 <<< dimGrid1, dimBlock1 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	
	dim3 dimGrid2(N2,1);
	dim3 dimBlock2(N1/2,1);
	kernelFunc_efficient2 <<< dimGrid2, dimBlock2 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	
	dim3 dimGrid3(N2,1);
	dim3 dimBlock3(N1/2,1);
	kernelFunc_efficient3 <<< dimGrid3, dimBlock3 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	*/
//----------------------------------------------------------------------------- 3D FFT 	(working for M<31)
	
	dim3 dimGrid4(N1*N1,1);
	dim3 dimBlock4(N3/2,1);
	kernelFunc_efficient4 <<< dimGrid4, dimBlock4 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	
	dim3 dimGrid5(N1*N3,1);
	dim3 dimBlock5(N1/2,1);
	kernelFunc_efficient5 <<< dimGrid5, dimBlock5 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	
	dim3 dimGrid6(N2,1);
	dim3 dimBlock6(N1/2,1);
	kernelFunc_efficient6 <<< dimGrid6, dimBlock6 >>>(x_r_d, x_i_d, X_r_d, X_i_d, N, M);
	
}
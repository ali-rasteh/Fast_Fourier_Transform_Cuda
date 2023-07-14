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

__device__ int shuffled_index_cal (int index , int digits){
	int i;
	int out_index = 0;
	for (i=0 ; i<digits ; i++){
		out_index += (((index>>i)&(0x1))<<(digits-1-i)) & (1<<(digits-1-i));
	}
	return out_index;
}

__device__ int power2(int in){
	int i;
	int in_power2 = 1;
	for (i=0 ; i<in ; i++){
		in_power2*=2;
	}
	return in_power2;
}

__device__ int log2(int in){
	int in1 = in;
	int in_log2 = 0;
	while (in1!=1){
		in1/=2;
		in_log2+=1;
	}
	return in_log2;
}

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





//-------------------------------------------------------------------------------------------------------------------------------------- 3D FFT





//-----------------------------------------------------------------------------
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
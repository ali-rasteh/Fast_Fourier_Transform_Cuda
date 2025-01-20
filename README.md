# Fast Fourier Transform with CUDA

This project provides an implementation of the Fast Fourier Transform (FFT) using CUDA for parallel computation on NVIDIA GPUs. The project includes both simple and efficient kernel implementations to perform FFT on complex data.

## Project Structure

- `FFT/`
  - `cuda printf/`
    - `cuPrintf.cu`: Implementation of a custom printf function callable from within CUDA kernels.
    - `cuPrintf.cuh`: Header file supporting `cuPrintf.cu`.
    - `sm_11_atomic_functions.h`: Header file providing atomic functions for CUDA architecture 1.1 and above.
  - `fft.cu`: Contains the implementation of various GPU kernels for FFT.
  - `fft.h`: Header file for FFT-related functions and constants.
  - `fft_main.cu`: Main file that runs the FFT on CPU and GPU, compares results, and measures performance.
  - `gpuerrors.h`: Header file for handling CUDA errors.
  - `gputimer.h`: Header file providing a timer class for measuring GPU execution time.

## Usage

### Prerequisites
- CUDA-enabled NVIDIA GPU
- CUDA Toolkit installed
- C++ compiler

### Compilation
To compile the project, use the following command:
```sh
nvcc FFT/fft_main.cu -o fft -I./FFT
```

### Running the Program
To run the compiled program, use the following command:
```sh
./fft SIMPLE M
```
- `SIMPLE`: Set to `1` for simple kernel execution, `0` for efficient kernel execution.
- `M`: Log2 of the number of data points (must be between 0 and 25).

Example:
```sh
./fft 1 10
```
This command runs the FFT with simple kernel execution on \(2^{10}\) data points.

### Output
The program prints the following information to the console:
- Device name
- Execution time on CPU
- Execution time on GPU
- Execution time of GPU kernels
- Mean squared error (MSE) between CPU and GPU results

## License
This project includes code with the following licenses:
- `cuPrintf.cu` and `cuPrintf.cuh`: NVIDIA Corporation, subject to NVIDIA's end user license agreement (EULA).
- `sm_11_atomic_functions.h`: NVIDIA Corporation, provided "as is" without warranty.

Refer to the respective files for the complete license information.

## Acknowledgments
This project utilizes CUDA for parallel computation on NVIDIA GPUs and includes sample code provided by NVIDIA Corporation.

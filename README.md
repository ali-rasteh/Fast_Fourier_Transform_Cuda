# Fast_Fourier_Transform_Cuda

Fast Fourier Transform Implementation in Cuda

## Description

This repository contains an implementation of the Fast Fourier Transform (FFT) using CUDA. It leverages the parallel computing power of NVIDIA GPUs to perform FFT calculations efficiently.

## Folder Structure

- `src/` - Contains the source code files written in C and CUDA.
- `data/` - Contains any input data files used for testing the implementation.
- `docs/` - Documentation files for the project.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C compiler (like GCC)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ali-rasteh/Fast_Fourier_Transform_Cuda.git
   cd Fast_Fourier_Transform_Cuda
   ```

2. Compile the source code:
   ```bash
   nvcc -o fft src/fft.cu
   ```

## Usage

Run the compiled executable:
```bash
./fft
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

## License

This project is licensed under the MIT License.


# Gaussian Blur Filter Optimization (OpenMP & CUDA)

## Project Overview
This project focuses on optimizing the standard **Gaussian Blur** image processing algorithm using High-Performance Computing (HPC) techniques. Applying a convolution filter on high-resolution images is a computationally expensive task ($O(N \times M \times K^2)$). 

The goal of this project was to take a slow, single-threaded serial implementation and parallelize it across multiple CPU cores and GPU multiprocessors to achieve maximum speedup.

*(Note: The base serial code and the CImg library template were provided as a university assignment base. All parallelization strategies and memory management implementations are original work.)*

---

## Parallelization Strategies

To drastically reduce the execution time, the algorithm was parallelized using two different architectures:

### 1. CPU Parallelization using OpenMP (`OMP_Codes`)
In this approach, the workload was distributed across multiple CPU threads.
* **Where we parallelized:** The outer `for` loops responsible for iterating over the image's width and height were parallelized using `#pragma omp parallel for`.
* **Impact:** This allowed different sections of the image to be processed simultaneously by different CPU cores, significantly reducing the bottleneck of serial pixel-by-pixel convolution.

### 2. GPU Parallelization using CUDA (`CudaCode`)
The ultimate performance gain was achieved by offloading the mathematical computations to the GPU.
* **Memory Flattening (Crucial Step):** Using 3D nested arrays on a GPU leads to severe memory latency and lack of coalescing. To solve this, the 3D image arrays were **flattened into 1D linear arrays** before being transferred to the GPU VRAM via `cudaMemcpy`.
* **Thread Allocation:** The nested loops were completely removed. Instead, each individual pixel's calculation was assigned to an independent CUDA thread.
* **Grid & Block Configuration:** The execution was configured using a `dim3(16, 16)` block size, providing 256 threads per block (a perfect multiple of NVIDIA's Warp Size of 32), ensuring optimal memory access and parallel efficiency.

---

## Results & Performance
By migrating from the serial CPU implementation to the highly parallelized CUDA architecture, the execution time experienced a massive speedup.

* **CUDA Execution Time:** The mathematical convolution operations on the GPU were completed in just **0.918 seconds** on the Tesla T4.
* **Conclusion:** The time recorded includes not only the mathematical operations within the GPU cores but also the relatively high overhead of transferring the flattened image data between the system RAM and the GPU VRAM. Despite this data transfer bottleneck, the massive parallelism of CUDA cores easily compensated for the latency, proving the architecture's dominance in image processing tasks.

---

## How to Run and Reproduce

### 1. CPU Execution (OpenMP)

#### Prerequisites (Windows Users)
To compile and run the OpenMP version on Windows, you need a C++ compiler with OpenMP support. We recommend using **MSYS2 (MinGW-w64)**:
1. Download and install [MSYS2](https://www.msys2.org/).
2. Install the GCC toolchain via MSYS2 terminal: `pacman -S mingw-w64-x86_64-gcc`
3. Add the MinGW binary path (usually `C:\msys64\mingw64\bin`) to your Windows **Environment Variables**.

#### Compilation & Execution
Once the compiler is set up, open your terminal in the project directory and run:

```bash
# Compile the code with the OpenMP flag enabled
g++ -fopenmp -O2 OMP_Codes/omp_code.cpp -o blur_omp -lgdi32

# Run the executable
./blur_omp
```
**(Note: If you are on Linux, you might need to link X11 and pthread libraries for CImg: -lX11 -lpthread)**

### 2. GPU Execution (CUDA on Google Colab)
Since this project handles heavy image arrays, executing the CUDA version on Google Colab is highly recommended to avoid local hardware constraints.

Upload the CudaCode folder, the cimg library, and the images folder to your Colab workspace.

Open a new Colab Notebook and set the Runtime type to GPU (e.g., T4).

Run the following commands in a notebook cell:

```Bash
# Compile the CUDA code
!nvcc CudaCode/cuda_code.cu -o blur_cuda

# Execute the compiled binary
!./blur_cuda
```

Tech Stack: C++, OpenMP, CUDA, Google Colab, Image Processing
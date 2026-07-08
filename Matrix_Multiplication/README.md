# Matrix Multiplication Optimization (OpenMP & CUDA)

## Project Overview
Matrix multiplication is a fundamental operation in scientific computing, machine learning, and deep learning. However, the standard mathematical approach has a time complexity of O(N × M × K), making it highly inefficient for large-scale matrices on a single CPU core.

The goal of this project was to parallelize the multiplication of two **non-square** matrices A[n × m] and B[m × k] into a resultant matrix C[n × k], ensuring maximum hardware utilization using both Multi-core CPUs (OpenMP) and Many-core GPUs (CUDA).

---

## Parallelization Strategies

### 1. CPU Parallelization using OpenMP (`OMP_Codes`)
To optimize the execution on the CPU, the workload was distributed among multiple threads.
* **The `collapse(2)` Magic:** A common mistake in OpenMP is only parallelizing the outermost loop (rows). If the number of rows is small but the columns are massive, most CPU cores will remain idle. To fix this, I utilized the `#pragma omp parallel for collapse(2)` directive.
* **Impact:** This collapses the two outer loops (rows and columns) into a single, massive linear iteration space. It ensures perfect **Load Balancing** across all available CPU threads, regardless of the matrix dimensions.

### 2. GPU Parallelization using CUDA (`CudaCode`)
To achieve massive speedup, the heavy mathematical calculations were offloaded to the GPU.
* **Memory Flattening (1D Arrays):** Dynamic 2D arrays (pointers to pointers) cause fragmented memory allocation, which is disastrous for GPU VRAM. Before transferring data via `cudaMemcpy`, the 2D matrices were flattened into **contiguous 1D arrays**. Elements were accessed mathematically using the formula: `Index = row * width + col`.
* **Thread Mapping:** Each element of the output matrix $C[n][k]$ was assigned to an entirely independent CUDA thread.
* **Non-Square Boundary Checks:** Since matrices $A$ and $B$ can have different dimensions, strict boundary conditions (`if (row < n && col < k)`) were implemented inside the CUDA Kernel to prevent out-of-bounds memory access (Segmentation Faults).
* **Grid & Block Configuration:** The execution was structured using 2D blocks `dim3(16, 16)` (256 threads per block) to perfectly align with NVIDIA's Warp architecture.

---

## How to Run and Reproduce

### 1. CPU Execution (OpenMP)

#### Prerequisites (Windows Users)
To compile and run the OpenMP version on Windows, we recommend using **MSYS2 (MinGW-w64)**:
1. Download and install [MSYS2](https://www.msys2.org/).
2. Install the GCC toolchain via MSYS2 terminal: `pacman -S mingw-w64-x86_64-gcc`
3. Add the MinGW binary path to your Windows **Environment Variables**.

#### Compilation & Execution
Open your terminal in the project directory and run:

```bash
# Compile the code with the OpenMP flag enabled
g++ -fopenmp -O2 OMP_Codes/matrix_omp.cpp -o matmul_omp

# Run the executable
./matmul_omp
```

### 2. GPU Execution (CUDA on Google Colab)
For evaluating the true power of GPU parallelism without local hardware constraints, executing the code on Google Colab is recommended.

Upload the CudaCode folder to your Colab workspace.

Open a new Colab Notebook and set the Runtime type to GPU (e.g., Tesla T4).

Run the following commands in a notebook cell:

```Bash
# Compile the CUDA code
!nvcc CudaCode/matrix_cuda.cu -o matmul_cuda

# Execute the compiled binary
!./matmul_cuda
```

Tech Stack: C++, OpenMP, CUDA, Linear Algebra, High-Performance Computing
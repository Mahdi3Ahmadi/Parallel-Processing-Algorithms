# High-Performance Computing (HPC) & Parallel Processing Portfolio

## Overview
Welcome to my High-Performance Computing portfolio. This repository contains a collection of computationally heavy algorithms that I have entirely re-architected and optimized to break the limitations of serial, single-threaded execution. 

The core focus of these projects is to utilize **Multi-core CPUs (via OpenMP)** and **Many-core GPUs (via NVIDIA CUDA C++)** to achieve massive speedups, solve memory bottlenecks, and implement complex data structures on hardware accelerators.

*(Note: The base serial implementations and visualization templates—such as the CImg library—were provided as university course materials. However, all parallelization architectures, memory flattening techniques, synchronization locks, and hybrid CPU-GPU designs were independently implemented.)*

---

## Projects Portfolio

### [Gaussian Blur Filter Optimization](./Gaussian_Blur_Filter)
An image processing project that accelerates the $O(N \times M \times K^2)$ convolution operations.
* **Key Achievements:** Eliminated deep nested arrays by implementing **Memory Flattening** (3D to 1D) for optimal GPU Memory Coalescing. Mapped pixel coordinates mathematically to a 2D CUDA Grid/Block architecture (`dim3(16, 16)`).

### [Matrix Multiplication (Non-Square)](./Matrix_Multiplication)
A robust linear algebra optimizer designed to handle the multiplication of large matrices with independent dimensions ($N \times M$ and $M \times K$).
* **Key Achievements:** Achieved perfect CPU thread load-balancing using OpenMP's `#pragma omp parallel for collapse(2)`. Prevented GPU segmentation faults through strict boundary condition checks inside the CUDA Kernels.

### [Hybrid N-Body Simulation (Barnes-Hut)](./Hybrid_NBody_BarnesHut)
**Featured Project:** A highly complex physics engine simulating gravitational forces using an $O(N \log N)$ Octree algorithm, dynamically balancing the workload between the CPU and GPU at every timestep.
* **Key Achievements:** * **CPU (OpenMP):** Multithreaded Octree construction using `omp_lock_t` (Atomic Locks) to prevent data race conditions without bottlenecking the system.
  * **GPU (CUDA C++):** Overcame the GPU's lack of support for deep recursion by designing a **Custom Stack-based Traversal** (`while` loop with a local array stack) to navigate the Octree purely on the graphic cores.

---

## Technology Stack
* **Languages:** `C++`, `CUDA C`
* **Parallel APIs:** `OpenMP`, `NVIDIA CUDA Toolkit`
* **Core Concepts:** Hybrid Architecture, Memory Coalescing, Thread Synchronization, Atomic Operations, Spatial Data Structures (Octree).
* **Environment:** `Google Colab (Tesla T4 GPU)`, `Linux / MSYS2 (MinGW-w64)`

---

## Navigation & Execution
Each project is encapsulated in its own directory with a dedicated, detailed `README.md` file. Please navigate to the specific project folders to view the source code, performance benchmarks, and instructions on how to compile and run the parallelized binaries.
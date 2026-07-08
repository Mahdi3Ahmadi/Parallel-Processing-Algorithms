# Hybrid N-Body Simulation (Barnes-Hut Algorithm)

## Project Overview
Simulating the gravitational forces between thousands of celestial bodies (N-Body problem) is a computationally massive task. The naive approach calculates every pair's interaction, resulting in an $O(N^2)$ time complexity. 

To overcome this, I implemented the **Barnes-Hut Algorithm**, which groups distant bodies into center-of-mass nodes using an **Octree** data structure, reducing the complexity to $O(N \log N)$. 

The true challenge and achievement of this project lie in its **Hybrid Parallel Architecture**. The workload is dynamically shared between Multi-core CPUs (via OpenMP) and Many-core GPUs (via CUDA) at every single timestep of the simulation.

---

## Hybrid Architecture & Parallelization Strategies

### Phase 1: Octree Construction on CPU (OpenMP)
Building an Octree requires dynamic memory allocation and complex branching, which GPUs handle poorly. Therefore, the tree construction was assigned to the CPU.
* **Multithreaded Insertion:** Bodies are inserted into the Octree concurrently using `#pragma omp parallel for`.
* **Atomic Locks (`omp_lock_t`):** To prevent race conditions when multiple threads attempt to write to the same Octree node, strict atomic locks were implemented at the node level, ensuring data integrity without locking the entire tree.

### Phase 2: Force Evaluation on GPU (CUDA)
Once the CPU builds the tree, the entire structure is flattened and transferred to the GPU VRAM. Each CUDA thread is assigned one body to calculate the net gravitational force acting upon it.
* **Stack-Based Traversal (No Recursion!):** GPU architectures do not support deep call stacks for recursive functions. To traverse the Octree on the GPU, I implemented a **custom local Stack array** (`int stack[64]`). Each thread independently navigates the tree using a `while(top > 0)` loop, pushing and popping nodes.
* **Multipole Acceptance Criterion (MAC):** During traversal, threads calculate the ratio of node size to distance ($s / d$). If the ratio is less than $\theta = 0.5$, the thread approximates the force using the node's center of mass and skips its children, saving massive amounts of computational time.

---

## Results & Performance
The simulation was executed for 10,000 bodies over multiple timesteps on an **NVIDIA Tesla T4** GPU (Google Colab).

* **CPU Tree Construction:** ~5.1 ms per timestep.
* **GPU Force Calculation & VRAM Transfer:** ~5.5 ms per timestep.
* **Conclusion:** Despite the PCIe bandwidth bottleneck caused by transferring the Octree between system RAM and VRAM at every timestep, the extreme mathematical throughput of the CUDA cores easily compensated for the delay. The result is a highly stable, ultra-fast physical simulation engine.

---

## How to Run and Reproduce

Since this project requires both OpenMP and CUDA toolchains to compile a hybrid binary, executing it on **Google Colab** is the standard approach.

1. Upload the `src` folder (containing the `.cu` files) to your Colab workspace.
2. Open a new Colab Notebook and set the Runtime type to **GPU** (e.g., T4).
3. Run the following commands in a notebook cell to compile the hybrid code:

```bash
# Compile the CUDA code with OpenMP flags enabled
!nvcc -Xcompiler -fopenmp src/hybrid_nbody.cu -o nbody_hybrid

# Execute the compiled binary
!./nbody_hybrid
```

Tech Stack: C++, CUDA C, OpenMP, Hybrid HPC, Data Structures (Octree), Physics Simulation
//%%writefile matrix_mul.cu
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda.h>

using namespace std;


__global__ void matrixMulKernel(double *d_A, double *d_B, double *d_C, int n, int m, int k) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (row < n && col < k) {
        double sum = 0.0;
        for (int l = 0; l < m; l++) {
            
            sum += d_A[row * m + l] * d_B[l * k + col];
        }
        d_C[row * k + col] = sum;
    }
}

int main() {
    
    int n = 1000;
    int m = 1200;
    int k = 800;

    cout << "CUDA Matrix Multiplication Test" << endl;
    cout << "A[" << n << "][" << m << "] * B[" << m << "][" << k << "] = C[" << n << "][" << k << "]\n";

    
    size_t size_A = n * m * sizeof(double);
    size_t size_B = m * k * sizeof(double);
    size_t size_C = n * k * sizeof(double);

    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);

    
    for (int i = 0; i < n * m; i++) h_A[i] = rand() % 100 / 10.0;
    for (int i = 0; i < m * k; i++) h_B[i] = rand() % 100 / 10.0;

    
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cout << "------------------------------------------\n";
    cout << "Launching CUDA Kernel with Grid(" << numBlocks.x << ", " << numBlocks.y << ") and Block(16, 16)...\n";

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n, m, k);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cout << "CUDA Execution Time (Kernel Only): " << milliseconds / 1000.0 << " seconds" << endl;
    cout << "------------------------------------------\n";

    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
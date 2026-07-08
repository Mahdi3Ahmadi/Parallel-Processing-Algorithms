//%%writefile GaussianBlurFilter.cu
#include <iostream>
#include <cuda.h>
#include "GaussianBlurFilter.h"


__global__ void gaussianBlurKernel(double *d_image, double *d_filter, double *d_newImage, 
                                   int width, int height, int filterWidth, int filterHeight, 
                                   int newImageWidth, int newImageHeight) 
{
   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (i < newImageWidth && j < newImageHeight) 
    {
        
        for (int d = 0; d < 3; d++) 
        {
            double sum = 0.0;
            for (int h = 0; h < filterWidth; h++) 
            {
                for (int w = 0; w < filterHeight; w++) 
                {
                    int imgX = i + h;
                    int imgY = j + w;
                    
                    
                    int imgIdx = d * (width * height) + imgX * height + imgY;
                    int filterIdx = h * filterHeight + w;
                    
                    sum += d_filter[filterIdx] * d_image[imgIdx];
                }
            }
            int outIdx = d * (newImageWidth * newImageHeight) + i * newImageHeight + j;
            d_newImage[outIdx] = sum;
        }   
    }
}

double **GBFilter::getGaussian(int height, int width, double sigma)
{
    double **filter;
    double sum = 0.0;
    int i, j;

    filter = new double *[height];
    for (int i = 0; i < height; i++)
        filter[i] = new double[width];

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
        {
            filter[i][j] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += filter[i][j];
        }

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
            filter[i][j] /= sum;

    return filter;
}

double ***GBFilter::applyFilter(double ***image, double **filter, int width, int height, int filterWidth, int filterHeight)
{
    int newImageHeight = height - filterHeight + 1;
    int newImageWidth = width - filterWidth + 1;

    
    int imgCells = 3 * width * height;
    double *h_flatImage = new double[imgCells];
    for (int d = 0; d < 3; d++)
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++)
                h_flatImage[d * (width * height) + i * height + j] = image[d][i][j];

    
    int filterCells = filterWidth * filterHeight;
    double *h_flatFilter = new double[filterCells];
    for (int i = 0; i < filterWidth; i++)
        for (int j = 0; j < filterHeight; j++)
            h_flatFilter[i * filterHeight + j] = filter[i][j];

    
    int outCells = 3 * newImageWidth * newImageHeight;
    double *h_flatOut = new double[outCells];

   
    double *d_image, *d_filter, *d_newImage;
    cudaMalloc((void**)&d_image, imgCells * sizeof(double));
    cudaMalloc((void**)&d_filter, filterCells * sizeof(double));
    cudaMalloc((void**)&d_newImage, outCells * sizeof(double));

    
    cudaMemcpy(d_image, h_flatImage, imgCells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_flatFilter, filterCells * sizeof(double), cudaMemcpyHostToDevice);

    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((newImageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (newImageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    
    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_image, d_filter, d_newImage, 
                                                       width, height, filterWidth, filterHeight, 
                                                       newImageWidth, newImageHeight);
    
    
    cudaDeviceSynchronize();

    
    cudaMemcpy(h_flatOut, d_newImage, outCells * sizeof(double), cudaMemcpyDeviceToHost);

    
    double ***newImage = new double **[3];
    for (int i = 0; i < 3; i++)
    {
        newImage[i] = new double *[width];
        for (int j = 0; j < width; j++)
        {
            newImage[i][j] = new double[height];
            for (int k = 0; k < height; k++) 
                newImage[i][j][k] = 0.0; 
        }
    }

    for (int d = 0; d < 3; d++)
        for (int i = 0; i < newImageWidth; i++)
            for (int j = 0; j < newImageHeight; j++)
                newImage[d][i][j] = h_flatOut[d * (newImageWidth * newImageHeight) + i * newImageHeight + j];

    
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_newImage);
    delete[] h_flatImage;
    delete[] h_flatFilter;
    delete[] h_flatOut;

    return newImage;
}
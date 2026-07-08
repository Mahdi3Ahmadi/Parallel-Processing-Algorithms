#ifdef _WIN32
#include "cimg\CImg.h"
#endif
#ifdef linux
#include "cimg/CImg.h"
#endif
#include <iostream>
#include <cmath>
#define M_PI 3.14159265358979323846

#include <omp.h>

using namespace cimg_library;
using namespace std;

double **getGaussian(int height, int width, double sigma)
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


double ***applyFilter(double ***image, double **filter, int width, int height, int filterWidth, int filterHeight)
{

    int newImageHeight = height - filterHeight + 1;
    int newImageWidth = width - filterWidth + 1;
    //int d, i, j, h, w;

    double ***newImage;
    newImage = new double **[3];
    for (int i = 0; i < 3; i++)
    {
        newImage[i] = new double *[newImageWidth];
        for (int j = 0; j < newImageWidth; j++)
            newImage[i][j] = new double[newImageHeight];
    }
#pragma omp parallel for collapse(3)
    for (int d = 0; d < 3; d++)
        for (int i = 0; i < newImageWidth; i++)
            for (int j = 0; j < newImageHeight; j++)
            {
                
                double pixel_sum = 0.0;

                for (int h = i; h < i + filterWidth; h++)
                    for (int w = j; w < j + filterHeight; w++)
                        pixel_sum += filter[h - i][w - j] * image[d][h][w];
                newImage[d][i][j] = pixel_sum;
            }

    return newImage;
}

int main(int argc, char const *argv[])
{
    // Read input image
    CImg<unsigned char> originalImage("img/input.jpg");
    /*
      For store RGB image data int  3d double array
      kernel[k][x][y]
      Dimension 1 (K): for Red Blue Green (RGB) spectrum
      Dimension 2 (x): pixels of image in width
      Dimension 3 (y): pixels of image in height
    */
    double ***kernel;

    int width = originalImage.width();
    int height = originalImage.height();

    // Memory Allocation for kernel
    kernel = new double **[3];
    for (int i = 0; i < 3; i++)
    {
        kernel[i] = new double *[width];
        for (int j = 0; j < width; j++)
            kernel[i][j] = new double[height];
    }

    // Copy input image to kernel
    for (int k = 0; k < 3; k++)
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                kernel[k][x][y] = originalImage(x, y, 0, k);

    // Showing orginal input image
    CImgDisplay Original_image(originalImage, "Original image");

    /*---*/
    int filterHeight = 25;
    int filterWidth = 25;

    cout << "Analyzing getGaussian (Filter Creation) Time...\n";
    double start_gaussian = omp_get_wtime();
    
    double **gaussianFilter = getGaussian(filterHeight, filterWidth, 10.0);
    
    double end_gaussian = omp_get_wtime();
    cout << "Gaussian Filter Creation Time: " << end_gaussian - start_gaussian << " seconds\n";
    cout << "------------------------------------------\n";

    /*// Apply Gaussian blur filter on input image
    kernel = applyFilter(kernel, getGaussian(filterHeight, filterWidth, 10.0), width, height, filterWidth, filterHeight);
    /*---*/

    int bluredImageHeight = height - filterHeight + 1;
    int bluredImageWidth = width - filterWidth + 1;

    double ***final_blurred_kernel = nullptr;


    int max_num_threads = 8;

    cout << "Starting OpenMP Parallelization Test...\n";
    cout << "------------------------------------------\n";

    for (int t = 1; t <= max_num_threads; t++)
    {
        
        omp_set_num_threads(t);

        double start_t = omp_get_wtime();

        
        double ***blurred_kernel = applyFilter(kernel, gaussianFilter, width, height, filterWidth, filterHeight);

        double end_t = omp_get_wtime();

        cout << "Threads: " << t << " \t| Execution Time: " << end_t - start_t << " seconds" << endl;

        
        if (t == max_num_threads)
        {
            final_blurred_kernel = blurred_kernel;
        }
        else
        {
            
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < bluredImageWidth; j++)
                    delete[] blurred_kernel[i][j];
                delete[] blurred_kernel[i];
            }
            delete[] blurred_kernel;
        }
    }
    cout << "------------------------------------------\n";

    // Create CImg object for convert double kernel to RGB image
    CImg<unsigned char> image(bluredImageWidth, bluredImageHeight, 1, 3);

    // Copy blured image to image object
    for (int k = 0; k < 3; k++)
        for (int x = 0; x < bluredImageWidth; x++)
            for (int y = 0; y < bluredImageHeight; y++)
                image(x, y, 0, k) = final_blurred_kernel[k][x][y];

    image.normalize(0, 255);
    // Showing blured image
    CImgDisplay bluredImage(image, "Blured Image");
    while (!bluredImage.is_closed())
        bluredImage.wait();
    return 0;
}

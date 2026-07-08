//%%writefile main.cpp
#include <iostream>
#include "GaussianBlurFilter.h"
#include "../cimg/CImg.h"
#include <iostream>
#include <cmath>
#include <chrono> 

using namespace cimg_library;

int main(int argc, char const *argv[])
{
    GBFilter filter;
    //Read input image
    CImg<unsigned char> originalImage("../img/input.jpg"), visu(500, 400, 1, 3, 0);
    /*
      For store RGB image data int  3d double aaray
      kernel[k][x][y]
      Dimension 1 (K): for Red Blue Green (RGB) spectrum
      Dimension 2 (x): pixels of image in width
      Dimension 3 (y): pixels of image in height
    */
    double ***kernel;

    int width = originalImage.width();
    int height = originalImage.height();

    //Memory Allocation for kernel
    kernel = new double **[3];
    for (int i = 0; i < 3; i++)
    {
        kernel[i] = new double *[width];
        for (int j = 0; j < width; j++)
            kernel[i][j] = new double[height];
    }

    //Copy input image to kernel
    for (int k = 0; k < 3; k++)
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                kernel[k][x][y] = originalImage(x, y, 0, k);

    //Showing orginal input image
    // CImgDisplay Original_image(originalImage, "Original image");

    /**/
    int filterHeight = 25;
    int filterWidth = 25;

    
    auto start_time = std::chrono::high_resolution_clock::now();

    
    kernel = filter.applyFilter(kernel, filter.getGaussian(filterHeight, filterWidth, 10.0), width, height, filterWidth, filterHeight);
    
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "=========================================" << std::endl;
    std::cout << "CUDA Parallel Execution Time: " << elapsed.count() << " seconds." << std::endl;
    std::cout << "=========================================" << std::endl;

    /*//Apply Gaussian blur filter on input image
    kernel = filter.applyFilter(kernel, filter.getGaussian(filterHeight, filterWidth, 10.0), width, height, filterWidth, filterHeight);
    /**/

    int bluredImageHeight = height - filterHeight + 1;
    int bluredImageWidth = width - filterWidth + 1;
    //Create CImg object for convert double kernel to RGB image
    CImg<unsigned char> image(bluredImageWidth, bluredImageHeight, 1, 3);

    //Copy blured image to image object
    for (int k = 0; k < 3; k++)
        for (int x = 0; x < bluredImageWidth; x++)
            for (int y = 0; y < bluredImageHeight; y++)
                image(x, y, 0, k) = kernel[k][x][y];

    image.normalize(0, 255);
    image.save("output.jpg");
    //Showing blured image
    //CImgDisplay bluredImage(image, "Blured Image");
    //while (!bluredImage.is_closed())
    //    bluredImage.wait();
    return 0;
}

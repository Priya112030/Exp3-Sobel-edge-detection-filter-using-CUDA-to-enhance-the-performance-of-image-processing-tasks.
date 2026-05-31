# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>AIM:</h3>
<h3>ENTER YOUR NAME : PRIYA B</h3>
<h3>ENTER YOUR REGISTER NO : 212224230208</h3>
<h3>EX. NO</h3>
<h3>DATE26.05.26</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
%%writefile sobelEdgeDetectionFilter.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {

    int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

    int sumX = 0;
    int sumY = 0;

    for(int i=-1;i<=1;i++){
        for(int j=-1;j<=1;j++){

            unsigned char pixel =
            srcImage[(y+i)*width + (x+j)];

            sumX += pixel * Gx[i+1][j+1];
            sumY += pixel * Gy[i+1][j+1];
        }
    }

    int magnitude = sqrtf(sumX*sumX + sumY*sumY);

    magnitude = min(max(magnitude,0),255);

    dstImage[y*width + x] =
    (unsigned char)magnitude;
}

int main()
{
    Mat image = imread("/content/images.jpeg",
                       IMREAD_GRAYSCALE);

    if(image.empty())
    {
        printf("Image not found\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize =
        width * height * sizeof(unsigned char);

    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;

    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage,
               image.data,
               imageSize,
               cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);

    dim3 gridSize(ceil(width/16.0),
                  ceil(height/16.0));

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height);

    cudaMemcpy(h_outputImage,
               d_outputImage,
               imageSize,
               cudaMemcpyDeviceToHost);

    Mat outputImage(height,
                    width,
                    CV_8UC1,
                    h_outputImage);

    imwrite("output_sobel.jpeg", outputImage);

    printf("Sobel Edge Detection Completed\n");

    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
```

## OUTPUT:
<img width="236" height="314" alt="cat" src="https://github.com/user-attachments/assets/3d152771-6d6d-4c29-8758-c95936f30aef" />
<img width="322" height="333" alt="image" src="https://github.com/user-attachments/assets/411ed59a-88aa-4266-92ca-c2486a7c0221" />


## RESULT:
Thus Sobel Edge Detection Filter using CUDA was implemented and executed successfully.

Questions:

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:


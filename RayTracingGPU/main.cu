
#include <iostream>
#include <cstdint>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "vec3.cuh"
#include "color.cuh"

__global__ void colourImage(uint8_t* img) 
{
    int y = blockIdx.x;
    int x = threadIdx.x;
    auto pixel_color = color(double(x)/255, double(y)/255, 0);
    write_color(pixel_color, img, y, x, 256);
}

int main(void)
{
    // Image

    const int image_width = 256;
    const int image_height = 256;
    // Render

    uint8_t img[image_width*image_height*3];
    uint8_t* img_device;

    cudaMalloc(&img_device, image_width*image_height*3);

    colourImage<<<256, 256>>>(img_device);

    cudaMemcpy(img, img_device, image_width*image_height*3, cudaMemcpyDeviceToHost);

    stbi_write_png("../../Image Samples/colourTest2.png", 256, 256, 3, img, 256*3);
    
    return 0;
}
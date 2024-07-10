#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"
#include "interval.cuh"

#include <iostream>

using color = vec3;

inline __device__ double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}

__device__ void write_color(uint8_t* out, double* img, const int row, const int col, const int width) {
    int pos = row*width*3 + col*3;
    // auto r = fmin(0.999,pixel_color.x());
    // auto g = fmin(0.999,pixel_color.y());
    // auto b = fmin(0.999,pixel_color.z());

    // auto r = pixel_color.x();
    // auto g = pixel_color.y();
    // auto b = pixel_color.z();
    
    auto r = fmin(0.999,linear_to_gamma(img[pos]));
    auto g = fmin(0.999,linear_to_gamma(img[pos+1]));
    auto b = fmin(0.999,linear_to_gamma(img[pos+2]));

    // Translate the [0,1] component values to the byte range [0,255].
    uint8_t rbyte = uint8_t(256 * r);
    uint8_t gbyte = uint8_t(256 * g);
    uint8_t bbyte = uint8_t(256 * b);

    // Write the pixel color components.
    out[pos] = rbyte;
    out[pos+1] = gbyte;
    out[pos+2] = bbyte;
}

#endif
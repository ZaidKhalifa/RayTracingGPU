#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"

#include <iostream>

using color = vec3;

__device__ void write_color(const color& pixel_color, uint8_t* out, const int row, const int col, const int width) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0,1] component values to the byte range [0,255].
    uint8_t rbyte = uint8_t(255.999 * r);
    uint8_t gbyte = uint8_t(255.999 * g);
    uint8_t bbyte = uint8_t(255.999 * b);
    // uint8_t rbyte = uint8_t(min(255, int(255.999 * r)));
    // uint8_t gbyte = uint8_t(min(255, int(255.999 * g)));
    // uint8_t bbyte = uint8_t(min(255, int(255.999 * b)));

    // Write the pixel color components.
    int pos = row*width*3 + col*3;
    out[pos] = rbyte;
    out[pos+1] = gbyte;
    out[pos+2] = bbyte;
}

#endif
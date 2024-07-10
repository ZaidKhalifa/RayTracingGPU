#ifndef INTERVAL_CUH
#define INTERVAL_CUH

#include "rtutil.cuh"

#define infinity 0x7ff0000000000000
#define n_infinity 0xfff0000000000000 

class interval {
  public:
    double min, max;
    __host__ __device__ interval() : min(infinity), max(n_infinity) {} // Default interval is empty

    __host__ __device__ interval(double min, double max) : min(min), max(max) {}

    __host__ __device__ double size() const {
        return max - min;
    }

    __host__ __device__ bool contains(double x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(double x) const {
        return min < x && x < max;
    }

    __host__ __device__ double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    // static const interval empty, universe;
};

// const interval interval::empty    = interval(+infinity, -infinity);
// const interval interval::universe = interval(-infinity, +infinity);

#endif
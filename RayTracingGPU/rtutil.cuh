#ifndef RTUTIL_CUH
#define RTUTIL_CUH

#include <iostream>
#include <random>
#include <limits>
#include <curand.h>
#include <curand_kernel.h>

// Common Headers

#include "color.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

// C++ Std Usings

// using std::make_shared;
// using std::shared_ptr;
// using std::sqrt;

// Constants

// const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;
const double twopi = 6.283185307179586477;
const double halfpi = 1.57079632679489661925;
const double inv_pi = 0.31830988618379067153;
const double sqrt_inv_pi = 0.56418958354775628694;
const interval Intensity{0.0, 0.999999};
// __constant__ double d_infinity;
__constant__ double d_pi;
__constant__ double d_twopi;
__constant__ double d_halfpi;
__constant__ double d_inv_pi;
__constant__ double d_sqrt_inv_pi;

#define infinity 0x7ff0000000000000
#define n_infinity 0xfff0000000000000

void init_constants()
{
    // cudaMemcpyToSymbol(d_infinity, &infinity, sizeof(double));
    cudaMemcpyToSymbol(d_pi, &pi, sizeof(double));
    cudaMemcpyToSymbol(d_twopi, &twopi, sizeof(double));
    cudaMemcpyToSymbol(d_halfpi, &halfpi, sizeof(double));
    cudaMemcpyToSymbol(d_inv_pi, &inv_pi, sizeof(double));
    cudaMemcpyToSymbol(d_sqrt_inv_pi, &sqrt_inv_pi, sizeof(double));
}

// Utility Functions

inline __device__ __host__ double degrees_to_radians(double degrees) 
{
    return degrees * d_pi / 180.0;
}

inline __device__ double random_double(curandState& state) {
    return 1.0 - curand_uniform_double(&state);
}

inline __device__ double random_double(curandState& state, double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double(state);
}

inline __host__ double random_double_h(double min = 0.0, double max = 1.0) {
    // Returns a random real in [min,max).
    static std::uniform_real_distribution<double> distribution(min, max);
    static std::mt19937 generator;
    return distribution(generator);
}

inline __device__ vec3 random_unit_vector(curandState& state) 
{
    double phi = random_double(state, 0, d_twopi);
    double theta = acos(random_double(state, -1, 1));
    return vec3(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
}

inline __device__ vec3 sample_unit_disk(curandState& state)
{
    double r = sqrt(random_double(state));
    double d = random_double(state, 0.0, d_twopi);
    return vec3(r*cos(d), r*sin(d), 0);
}

inline __device__ vec3 random_on_hemisphere(curandState& state, const vec3& normal) 
{
    vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

inline __host__ vec3 color_random(double min = 0.0, double max = 1.0) 
    {
        return vec3(random_double_h(min,max), random_double_h(min,max), random_double_h(min,max));
    }

#endif
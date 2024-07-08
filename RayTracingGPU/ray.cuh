#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class ray {
  public:
    __device__ __host__ ray() {}

    __device__ __host__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    __device__ __host__ const point3& origin() const  { return orig; }
    __device__ __host__ const vec3& direction() const { return dir; }

    __device__ __host__ point3 at(double t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    vec3 dir;
};

#endif
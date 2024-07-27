#ifndef HITBOX_CUH
#define HITBOX_CUH

#include "ray.cuh"

class material;

class hit_record {
  public:
    point3 p;
    vec3 normal;
    material* mat;
    double t;
    bool front_face;

    __device__ __host__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hitbox {
  public:
    // __host__ __device__ virtual ~hitbox() = default;
    __host__ __device__ virtual ~hitbox(){};

    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

    // __device__ virtual void printSphere()
    // {
    //     printf("pipii");
    // }
};

#endif
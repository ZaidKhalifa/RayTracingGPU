#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "hitbox.cuh"
#include "ray.cuh"
#include "rtutil.cuh"

class hit_record;

class material {
  public:
    __device__ virtual ~material() = default;

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& state) const 
    {
        return false;
    }
};

class lambertian : public material {
  public:
    __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& state)
    const override {
        vec3 scatter_direction = rec.normal + random_unit_vector(state);

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

  private:
    color albedo;
};

class metal : public material {
  public:
    __device__ metal(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& state)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return true;
    }

  private:
    color albedo;
};

#endif
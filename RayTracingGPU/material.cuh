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
    __device__ metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz: 1) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& state)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(state));
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

  private:
    color albedo;
    double fuzz;
};

class dielectric : public material {
  public:
    __device__ dielectric(double refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        vec3 refracted = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, refracted);
        return true;
    }

  private:
    // It's actually the relative index, not the absolute index
    double refraction_index;
};

#endif
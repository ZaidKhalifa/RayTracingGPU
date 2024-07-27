#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "rtutil.cuh"
#include "hitbox.cuh"
#include "vec3.cuh"

class sphere : public hitbox {
public:
    __host__ __device__ sphere(const point3& center, double radius, material* mat) : center(center), radius(fmax(0.0,radius)), mat(mat) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

    // __device__ void printSphere()
    // {
    //     printf("Center = (%lf, %lf, %lf)\nRadius = %lf\n", center[0], center[1], center[2], radius);
    // }

private:
    point3 center;
    double radius;
    material* mat;
};

#endif
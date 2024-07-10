#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "rtutil.cuh"

#include "hitbox.cuh"


class camera {
public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int image_width = 100;      // Rendered image width in pixel count
    int max_depth = 50;         // Maximum number of ray bounces into scene

    __device__ void init(const double aspectRatio = 1.0, const int imgWidth = 100)
    {
        aspect_ratio = aspectRatio;
        image_width = imgWidth;
        initialize();
    }

    __device__ color render(const hitbox* world, const int& x, const int& y, curandState& state) {

        ray r = get_ray(state, x, y);

        return ray_color(r, max_depth, world, state);
    }

private:
    int    image_height;   // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below

    __device__ void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = point3(0, 0, 0);

        // Determine viewport dimensions.
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    __device__ ray get_ray(curandState& state, int x, int y) const {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        auto offset = sample_unit_disk(state);
        auto pixel_sample = pixel00_loc + ((x + offset.x()) * pixel_delta_u) + ((y + offset.y()) * pixel_delta_v);

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    // __device__ color ray_color(const ray& r, int depth, const hitbox* world, curandState& state) const {
    //     if (depth <= 0)
    //         return color(0,0,0);

    //     hit_record rec;

    //     if (world->hit(r, interval(0.001, infinity), rec)) 
    //     {
    //         vec3 direction = random_on_hemisphere(state, rec.normal);
    //         return 0.5 * ray_color(ray(rec.p, direction), depth-1, world, state);
    //     }

    //     vec3 unit_direction = unit_vector(r.direction());
    //     auto a = 0.5*(unit_direction.y() + 1.0);
    //     return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    // }

    __device__ color ray_color(ray r, int depth, const hitbox* world, curandState& state) const {
        double intensity_so_far = 1.00000000;
        for(int i = 0; i < max_depth; i++)
        {
            hit_record rec;

            if (world->hit(r, interval(0.001, infinity), rec)) 
            {
                vec3 direction = rec.normal + random_unit_vector(state);
                intensity_so_far *= 0.500000000;
                r = ray(rec.p, direction);
            }
            else
            {
                vec3 unit_direction = unit_vector(r.direction());
                auto a = 0.5*(unit_direction.y() + 1.0);
                return intensity_so_far*((1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0));
            }
        }
        return color(0,0,0);
    }
};

#endif
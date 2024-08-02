#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "rtutil.cuh"

#include "hitbox.cuh"
#include "material.cuh"


class camera {
public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int image_width = 100;      // Rendered image width in pixel count
    int max_depth = 50;         // Maximum number of ray bounces into scene

    double vfov = 90;                   // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);    // Point camera is looking from
    point3 lookat   = point3(0,0,-1);   // Point camera is looking at
    vec3   vup      = vec3(0,1,0);      // Camera-relative "up" direction

    double defocus_angle = 0;       // Variation angle of rays through each pixel
    double focus_dist    = 10;      // Distance from camera lookfrom point to plane of perfect focus

    __device__ void init(const double aspectRatio, const int imgWidth, const int vFov, const point3 lookFrom, const point3 lookAt, const vec3 vUp, const double defocusAngle, const double focusDist)
    {
        aspect_ratio = aspectRatio;
        image_width = imgWidth;
        vfov = vFov;
        lookfrom = lookFrom;
        lookat = lookAt;
        defocus_angle = defocusAngle;
        focus_dist = focusDist;
        initialize();
    }

    __device__ color render(const hitbox* world, const int& x, const int& y, curandState& state) {

        ray r = get_ray(state, x, y);

        return ray_color(r, max_depth, world, state);
    }

private:
    int    image_height;    // Rendered image height
    point3 center;          // Camera center
    point3 pixel00_loc;     // Location of pixel 0, 0
    vec3   pixel_delta_u;   // Offset to pixel to the right
    vec3   pixel_delta_v;   // Offset to pixel below
    vec3   u, v, w;         // Camera frame basis vectors
    vec3   defocus_disk_u;  // Defocus disk horizontal radius
    vec3   defocus_disk_v;  // Defocus disk vertical radius

    __device__ void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2.0);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = viewport_width * u;
        auto viewport_v = viewport_height * -v;

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - focus_dist*w - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2.0));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ ray get_ray(curandState& state, int x, int y) const {
        // Construct a camera ray originating from the defocus disk and directed at randomly sampled
        // point around the pixel location i, j.

        auto offset = sample_unit_disk(state);
        auto pixel_sample = pixel00_loc + ((x + offset.x()) * pixel_delta_u) + ((y + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(state);
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    __device__ point3 defocus_disk_sample(curandState& state) const {
        // Returns a random point in the camera defocus disk.
        auto p = sample_unit_disk(state);
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    __device__ color ray_color(ray r, int depth, const hitbox* world, curandState& state) const {
        color intensity_so_far = color(1.0, 1.0, 1.0);
        for(int i = 0; i < max_depth; i++)
        {
            hit_record rec;

            if (world->hit(r, interval(0.001, infinity), rec)) 
            {   
                ray scattered;
                color attenuation;
                if (rec.mat->scatter(r, rec, attenuation, scattered, state))
                    intensity_so_far *= attenuation;
                r = scattered;
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
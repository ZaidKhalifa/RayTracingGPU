#include <cstdint>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "rtutil.cuh"

#include "hitbox.cuh"
#include "hitbox_list.cuh"
#include "sphere.cuh"


__device__ double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = h*h - a*c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - sqrt(discriminant) ) / (2.0*a);
    }
}

__device__ color ray_color(const ray& r, hitbox* world) {
    hit_record rec;
    if (world->hit(r, 0, d_infinity, rec)) 
    {
        return 0.5 * (rec.normal + color(1,1,1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

// __device__ color ray_color(const ray& r, sphere* world) {
//     hit_record rec;
//     auto t = world->hit(r, 0, d_infinity, rec);
//     // printf("ffff\n");
//     if (t > 0.0) {
//         vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
//         return 0.5*color(N.x()+1, N.y()+1, N.z()+1);
//     }

//     vec3 unit_direction = unit_vector(r.direction());
//     auto a = 0.5*(unit_direction.y() + 1.0);
//     return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
// }

__global__ void colourImage(uint8_t* img, const vec3 pixel00_loc, const vec3 pixel_delta_u, const vec3 pixel_delta_v, const vec3 camera_center, hitbox_list** world) 
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int image_width = gridDim.x;
    // int y = threadIdx.x;

    auto pixel_center = pixel00_loc + (x * pixel_delta_u) + (y * pixel_delta_v);
    auto ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);

    // color pixel_color = ray_color(r, world);
    // (*world)->printSphere();
    color pixel_color = ray_color(r, *world);
    write_color(pixel_color, img, y, x, image_width);
}

//Cuda functions

__global__ void initWorld(hitbox_list** world)
{
    (*world) = new hitbox_list();
}

__global__ void addSphere(hitbox_list** world, point3 center, double radius)
{
    (*world)->add(new sphere(center, radius));
}

int main(void)
{
    init_constants();

    // Image

    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 1280;
    // int image_width = 4;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = max(1,int(image_width / aspect_ratio));

    // World

    hitbox_list** world;
    cudaMalloc(&world, sizeof(hitbox_list*));
    initWorld<<<1,1>>>(world);
    addSphere<<<1,1>>>(world, point3(0,0,-1), 0.5);
    addSphere<<<1,1>>>(world, point3(0,-100.5,-1), 100);


    // Camera

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Render

    uint8_t* img;
    uint8_t* img_device;

    img = (uint8_t*) malloc(sizeof(uint8_t)*image_width*image_height*3);
    cudaMalloc(&img_device, image_width*image_height*3);


    colourImage<<<{(unsigned int)image_width, (unsigned int)image_height, 1}, 1>>>(img_device, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, world);

    cudaMemcpy(img, img_device, image_width*image_height*3, cudaMemcpyDeviceToHost);

    stbi_write_png("../../Image Samples/envTest.png", image_width, image_height, 3, img, image_width*3);
    
    return 0;
}
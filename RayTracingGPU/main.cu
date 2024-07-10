#include <cstdint>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "rtutil.cuh"

#include "camera.cuh"
#include "hitbox.cuh"
#include "hitbox_list.cuh"
#include "sphere.cuh"

__global__ void render(double* img, camera** cam, hitbox_list** world, int image_width, curandState *states, double frac, uint8_t* imgu) 
{
    int x = blockIdx.x%image_width;
    int y = blockIdx.y;
    int pixel_count = gridDim.y*image_width;
    // int image_width = gridDim.x;
    // int y = threadIdx.x;
    int b_id = image_width*y + x;
    curandState localState = states[(b_id+threadIdx.x)%pixel_count];

    color pixel_color = (*cam)->render(*world, x, y, localState);

    states[(b_id+threadIdx.x)%pixel_count] = localState;

    int pos = y*image_width*3 + x*3;

    atomicAdd(&img[pos], frac*pixel_color.x());
    atomicAdd(&img[pos+1], frac*pixel_color.y());
    atomicAdd(&img[pos+2], frac*pixel_color.z());

    // img[pos] = frac*pixel_color.x();
    // img[pos+1] = frac*pixel_color.y();
    // img[pos+2] = frac*pixel_color.z();

    // write_color(imgu, img, y, x, image_width);
    // write_color(pixel_color, imgu, y, x, image_width);
}

__global__ void make_image(double* img, uint8_t* imgu, int image_width) 
{
    int x = blockIdx.x;
    int y = blockIdx.y; 

    write_color(imgu, img, y, x, image_width);
}

//Cuda functions

__global__ void setup_random(curandState *states)
{
    unsigned long long id = gridDim.x*blockIdx.y + blockIdx.x;
    curand_init(id, 0, 0, &(states[id]));
    
    // printf("\n(%d, %d):\n",blockIdx.y,blockIdx.x);
    // for(int i = 0; i < 5; i++)
    // {
    //     curandState st = states[id];
    //     printf("%lf\n",curand_uniform_double(&st));
    // }
}

__global__ void createCam(camera** cam, double aspectRatio, int imgWidth)
{
    (*cam) = new camera();
    (*cam)->init(aspectRatio, imgWidth);
}

__global__ void initWorld(hitbox_list** world)
{
    (*world) = new hitbox_list();
}

__global__ void addSphere(hitbox_list** world, point3 center, double radius)
{
    (*world)->add(new sphere(center, radius));
}

__global__ void clean(camera** cam, hitbox_list** world)
{
    delete (*cam);
    (*world)->clear();
    delete (*world);
}

int main(void)
{
    init_constants();

    // Image

    auto aspect_ratio = 16.0 / 9.0;
    unsigned int image_width = 1920;
    // unsigned int image_width = 4;

    // Calculate the image height, and ensure that it's at least 1.
    unsigned int image_height = max(1,int(image_width / aspect_ratio));

    // World

    hitbox_list** world;
    cudaMalloc(&world, sizeof(hitbox_list*));
    initWorld<<<1,1>>>(world);
    addSphere<<<1,1>>>(world, point3(0,0,-1), 0.5);
    addSphere<<<1,1>>>(world, point3(0,-100.5,-1), 100);


    // Camera

    camera** cam;
    cudaMalloc(&cam, sizeof(camera*));
    createCam<<<1,1>>>(cam, aspect_ratio, image_width);

    // Render

    unsigned int samples = 2048;
    // unsigned int divisions = samples/512;
    unsigned int divisions = (samples+511)/512;
    // unsigned int rem = samples-divisions*512;
    unsigned int samples_per_block = samples/divisions;
    double frac = 1.0/(samples_per_block*divisions);
    // frac = 1.0;

    std::cout << "Number of samples are "<< samples_per_block*divisions <<".\n";

    uint8_t* img;
    uint8_t* img_device;
    double* img_doubles;
    curandState* rand_states;


    img = (uint8_t*) malloc(sizeof(uint8_t)*image_width*image_height*3);
    cudaMalloc(&img_device, image_width*image_height*3);
    cudaMalloc(&img_doubles, image_width*image_height*3*sizeof(double));
    cudaMemset(img_doubles, 0.0, image_width*image_height*3);
    cudaMalloc(&rand_states, image_width*image_height*sizeof(curandState));

    setup_random<<<{image_width, image_height, 1}, 1>>>(rand_states);

    cudaDeviceSynchronize();

    // for (int i = 0; i < divisions; i++)
    // {
    //     render<<<{image_width, image_height, 1}, 512>>>(img_doubles, cam, world, image_width, rand_states, frac, img_device);
    // }
    // render<<<{image_width, image_height, 1}, rem>>>(img_doubles, cam, world, image_width, rand_states, frac, img_device);

    render<<<{image_width, image_height, divisions}, samples_per_block>>>(img_doubles, cam, world, image_width, rand_states, frac, img_device);

    cudaDeviceSynchronize();

    make_image<<<{image_width, image_height, 1}, 1>>>(img_doubles, img_device, image_width);

    cudaMemcpy(img, img_device, image_width*image_height*3, cudaMemcpyDeviceToHost);

    stbi_write_png("../../Image Samples/envTest.png", image_width, image_height, 3, img, image_width*3);

    clean<<<1,1>>>(cam, world);
    cudaFree(world);
    cudaFree(cam);
    free(img);
    cudaFree(img_device);
    cudaFree(img_doubles);
    cudaFree(rand_states);
    
    return 0;
}
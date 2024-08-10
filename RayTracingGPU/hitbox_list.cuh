#ifndef HITBOX_LIST_CUH
#define HITBOX_LIST_CUH

#include "hitbox.cuh"

class hitbox_list : public hitbox {
  public:
    hitbox** objects;

    __host__ __device__ hitbox_list(int m_obj = 500)
    {
        max_obj = m_obj;
        obj_count = 0;
        objects = new hitbox*[max_obj];
    }

    // __host__ hitbox_list(int m_obj = 100, hitbox* object) 
    // { 
    //     max_obj = m_obj;
    //     obj_count = 0;
    //     objects = new hitbox*[max_obj];
    //     cudaMalloc((hitbox***)&d_objects, max_obj*sizeof(hitbox*));
    //     add(object); 
    // }

    __host__ __device__ void clear() 
    { 
        for(int i = 0; i < obj_count; i++)
        {
            delete objects[i];
        }
        obj_count = 0;
    }

    __host__ __device__ void add(hitbox* object) 
    {
        if(obj_count<max_obj)
            objects[obj_count++] = object;
        else
            printf("Max object limit of %d reached.\n", max_obj);
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < obj_count; i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __host__ __device__ ~hitbox_list() 
    { 
        clear();
        delete[] *objects;
    }
  private:
    int max_obj;
    int obj_count;
};

#endif
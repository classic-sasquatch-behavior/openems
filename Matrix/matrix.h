#pragma once


#include<vector>
#include<cuda.h>

#include"cuda_macros.h"

template<typename Type>
struct Device_Ptr{

    Type* data;

    int dims[3] = {1,1,1};
    int num_dims = 0;

    __device__ __host__ inline int height() const {return dims[0];}
    __device__ __host__ inline int width() const {return dims[1];}
    __device__ __host__ inline int depth() const {return dims[2];}

    __device__ __host__ inline int size(){return dims[0] * dims[1] * dims[2];}

    //constructor
    Device_Ptr(Matrix* parent){
        data = parent->device_data;
        num_dims = parent->num_dims;

        for(int i = 0; i < num_dims; i++){
            dims[i] = parent->dims[i];
        }
    }

    Type& operator()(int row, int col = 0, int cbc = 0){
        return data[(row * width() + col) * depth() + cbc];
    }

};


template<typename Type>
struct Matrix{

    Type* host_data;
    Type* device_data;

    int dims[3] = {1,1,1}
    int num_dims = 0;

    __device__ __host__ inline int height() const {return dims[0];}
    __device__ __host__ inline int width() const {return dims[1];}
    __device__ __host__ inline int depth() const {return dims[2];}

    __device__ __host__ inline int size(){return dims[0] * dims[1] * dims[2];}
    __device__ __host__ inline int bytesize(){return size() * sizeof(Type);}


    //-----data access-----

    Type& operator()(int row, int col = 0, int cbc = 0){
        return[(row * width() + col) * depth() + cbc];
    }

    //typecast to device
    operator Device_Ptr<Type>(){
        return Device_Ptr(this);
    }

    //constructor
    Matrix(std::vector<int>_dims){
        num_dims = dims.size();

        for(int i = 0; i < num_dims; i++){
            dims[i] = _dims[i];
        }

        allocate();
    }

    //-----rule of three-----

    //destructor
    ~Matrix(){
        delete host_data;
        cudaFree(device_data);
    }

    //copy constructor
    Matrix(const Matrix<Type>& input){
        num_dims = input.num_dims;

        for(int i = 0; i < num_dims; i++){
            dims[i] = input.dims[i];
        }

        allocate();
        load(input.host_data);
    }

    //copy assignment operator
    void operator=(const Matrix<Type>& input){
        num_dims = input.num_dims;

        for(int i = 0; i < num_dims; i++){
            dims[i] = input.dims[i];
        }

        load(input.host_data);
    }


    //-----memory-----
    void load(Type* input){
        
        //deep copy input data (which is assumed to be on the host)
        for(int i = 0; i < bytesize(); i++){
            host_data[i] = input[i];
        }

        //upload newly copied data to device
        upload();
    }

    void allocate(){
        //allocate host_data (dont forget to delete it!)
        host_data = new Type[bytesize()];

        //allocate device_data (this gets deleted with cudaFree())
        //cuda safe call will be a macro which performs error checking (see cuda_macros.h)
        CUDA_SAFE_CALL( cudaMalloc((void**)&device_data, bytesize()) );
    }

    void upload(){
        //copy memory from host(CPU) to device(GPU)
        cudaMemcpy(device_data, host_data, bytesize(), cudaMemcpyHostToDevice);
    }

    void download(){
        //copy memory from device(GPU) to host(CPU)
        cudaMemcpy(host_data, device_data, bytesize(), cudaMemcpyDeviceToHost);
    }


};
#pragma once


#include<vector>
#include<cuda.h>

#include"cuda_macros.h"


template<typename Type>
struct Matrix{

    Type* host_data;
    Type* device_data;

    int dims[3] = {1,1,1}
    int num_dims = 0;

    __device__ __host__ inline int size(){return dims[0] * dims[1] * dims[2];}
    __device__ __host__ inline int bytesize(){return size() * sizeof(Type);}

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
        for(int i = 0; i < bytesize(); i++){
            host_data[i] = input[i];
        }
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

    }

    void download(){

    }


};
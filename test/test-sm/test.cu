#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h> 
#include <stdio.h> 
// #define VECTOR_LENGTH 10000 
// #define MAX_ERR 1e-4
__global__ void pow_100(float* arr,int len){
    // int tid = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int temp_id = tid;
    
    // if(tid<len-1)
    // for(int i=0;i<1000000000;i++){
//    while(true){    
        arr[tid] *= arr[tid];
        // __syncthreads();
  //  }
}
__global__ void sqrt_100(float* arr,int len){
    // int tid = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int temp_id = tid;
    
    for(int i=0;i<10000;i++){
        arr[tid] = sqrtf(arr[tid]);
        // __syncthreads();
    }
}

void cuda_pow(float* arr,int N){ 
    float* device_arr;
    cudaMalloc(&device_arr,N*sizeof(float));
    cudaMemcpy(device_arr,arr,N*sizeof(float),cudaMemcpyHostToDevice);
    const int threadsPerBlock = 512;
    const int blocks = (N+ threadsPerBlock - 1 )/threadsPerBlock;
    pow_100<<<blocks,threadsPerBlock>>>(device_arr,threadsPerBlock);
    cudaDeviceSynchronize();
    cudaMemcpy(arr,device_arr,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(device_arr);
}

void cuda_sqrt(float* arr,int N){ 
    float* device_arr;
    cudaMalloc(&device_arr,N*sizeof(float));
    cudaMemcpy(device_arr,arr,N*sizeof(float),cudaMemcpyHostToDevice);
    const int threadsPerBlock = 1024;
    const int blocks = (N+ threadsPerBlock - 1 )/threadsPerBlock;
    sqrt_100<<<blocks,threadsPerBlock>>>(device_arr,threadsPerBlock);
    cudaDeviceSynchronize();
    cudaMemcpy(arr,device_arr,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(device_arr);
}



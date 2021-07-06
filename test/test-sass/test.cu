  
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>


__global__ void set_one(int * arr ){
  int tid = blockIdx.x*blockDim.x+ threadIdx.x;
  arr[tid] = 1;
}

int main(){
  const int N=10;
  int* device_arr;
  cudaMalloc(&device_arr,N*sizeof(int));
  // cudaMemcpy(device_arr,arr,N*sizeof(int),cudaMemcpyHostToDevice);
  set_one<<<1,N>>>(device_arr);
  cudaDeviceSynchronize();
  // cudaMemcpy(arr,device_arr,N*sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(device_arr);
  return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
void cuda_pow(float* arr,int N);
void cuda_sqrt(float* arr,int N);
const int N =100000;  
const int times=10000;
int main(){
    cudaSetDevice(3);
    float* arr = new float[N];
    for(int i=0;i<N;++i) arr[i]=1.0f;
    for(int i = 0 ; i< N ;++i){
        cuda_pow(arr,N);
        // cuda_sqrt(arr,N);
    }
    delete[] arr;
}

#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // --------------------------------------------------------------------------------
    // 1. allocate host memory
    // --------------------------------------------------------------------------------
    const int N = 1000;
    float A_h[N], B_h[N], C_h[N] = {0};
    std::srand(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < N; ++i)
    {
        A_h[i] = static_cast<float>(std::rand()) / RAND_MAX;
        B_h[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    int size = N * sizeof(float);

    // --------------------------------------------------------------------------------
    // 2. allocate device memory
    // --------------------------------------------------------------------------------
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // --------------------------------------------------------------------------------
    // 3. copy host to device
    // --------------------------------------------------------------------------------
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // --------------------------------------------------------------------------------
    // 4. launch kernel
    // --------------------------------------------------------------------------------
    int block_dim = 256;
    int grid_dim = int(N / 256.0 + 0.5);
    vectorAddKernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, N);

    // --------------------------------------------------------------------------------
    // 5. copy device to host
    // --------------------------------------------------------------------------------
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // --------------------------------------------------------------------------------
    // 6. free device memory
    // --------------------------------------------------------------------------------
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}

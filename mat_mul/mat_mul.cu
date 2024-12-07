#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define checkCudaErrors(func)                                                      \
    {                                                                              \
        (func);                                                                    \
        cudaError_t e = cudaGetLastError();                                        \
        if (e != cudaSuccess)                                                      \
        {                                                                          \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        }                                                                          \
    }

__global__ void NaiveMatMulKernel(const float *A, const float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float sum = 0;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template <size_t M, size_t K, size_t N, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__global__ void TiledMatMulKernel1(const float *A, const float *B, float *C)
{
    // 把 M 的连续多行和 N 的连续多列全放进 smem
    // 只能用小矩阵测试这个 kernel, 否则很容易将可能 48KB 用完 (4060ti)
    // smem 占用太多会报 CUDA Error: a PTX JIT compilation failed
    // 所以这种方式是行不通的
    __shared__ float tileA[BLOCK_HEIGHT][K];
    __shared__ float tileB[K][BLOCK_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // 假设 BLOCK_WIDTH/HEIGHT 和 blockDim.x/y 一样大
    int rowA = by * BLOCK_HEIGHT + ty; // 当前线程对应的 A/C 的行
    int colB = bx * BLOCK_WIDTH + tx;  // 当前线程对应的 B/C 的列

    // copy A to tileA
    for (int i = 0; i < (K + BLOCK_WIDTH - 1) / BLOCK_WIDTH; ++i)
    {
        int col = i * BLOCK_WIDTH + tx;
        if (rowA < M && col < K)
        {
            tileA[ty][col] = A[rowA * K + col];
        }
        else
        {
            tileA[ty][col] = 0.0f;
        }
    }

    // copy B to tileB
    for (int i = 0; i < (K + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT; ++i)
    {
        int row = i * BLOCK_HEIGHT + ty;
        if (colB < N && row < K)
        {
            tileB[row][tx] = B[row * N + colB];
        }
        else
        {
            tileB[row][tx] = 0.0f;
        }
    }

    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < K; ++i)
    {
        sum += tileA[ty][i] * tileB[i][tx];
    }

    if (rowA < M && colB < N)
    {
        C[rowA * N + colB] = sum;
    }
}

template <size_t M, size_t K, size_t N, size_t BLOCK_WIDTH, size_t BLOCK_HEIGHT>
__global__ void TiledMatMulKernel2(const float *A, const float *B, float *C)
{
    // TODO: 每次放 A 和 B 的一小部分到 smem 避免放不下
}

// row-major
float *CreateHostMatrix(size_t sizeInBytes, bool zeroInit)
{
    float *matrix = (float *)malloc(sizeInBytes);
    if (zeroInit)
    {
        memset(matrix, 0, sizeInBytes);
    }
    else
    {
        std::mt19937 gen(7355608);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < sizeInBytes / sizeof(float); ++i)
        {
            matrix[i] = dist(gen);
        }
    }
    return matrix;
}

bool CheckEqual(void *lhs, void *rhs, size_t nBytes)
{
    return memcmp(lhs, rhs, nBytes) == 0;
}

int main()
{
    constexpr size_t M = 16, K = 16, N = 16;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    float *h_A = CreateHostMatrix(sizeA, false);
    float *h_B = CreateHostMatrix(sizeB, false);
    float *h_C0 = CreateHostMatrix(sizeC, true);
    float *h_C1 = CreateHostMatrix(sizeC, true);
    float *h_C2 = CreateHostMatrix(sizeC, true);

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, sizeA));
    checkCudaErrors(cudaMalloc((void **)&d_B, sizeB));
    checkCudaErrors(cudaMalloc((void **)&d_C, sizeC));

    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    constexpr size_t blockSizeX = 16, blockSizeY = 16;
    dim3 dimBlock(blockSizeX, blockSizeY);
    dim3 dimGrid((N + blockSizeX - 1) / blockSizeX, (M + blockSizeY - 1) / blockSizeY);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 10;

    // --------------------------------------------------------------------------------
    // cuBlas or cutlass as reference
    // --------------------------------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 矩阵乘法 C = alpha * A * B + beta * C
    // A: M x K, B: K x N, C: M x N
    // cuBLAS 使用列优先存储，所以矩阵维度和布局是颠倒的
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // 不转置 A 和 B
        N,                        // B 的列数 (C 的列数)
        M,                        // A 的行数 (C 的行数)
        K,                        // A 的列数 (B 的行数)
        &alpha,                   // 标量 alpha
        d_B, N,                   // B 矩阵，列主序，leading dimension = N
        d_A, K,                   // A 矩阵，列主序，leading dimension = K
        &beta,                    // 标量 beta
        d_C, N                    // C 矩阵，列主序，leading dimension = N
    );

    checkCudaErrors(cudaMemcpy(h_C0, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --------------------------------------------------------------------------------
    // 1. naive matmul
    // --------------------------------------------------------------------------------

    checkCudaErrors(cudaEventRecord(start));

    for (size_t i = 0; i < nIter; ++i)
    {
        NaiveMatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("NaiveMatMulKernel time cost %f\n", msecTotal);

    checkCudaErrors(cudaMemcpy(h_C1, d_C, sizeC, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------------------------------
    // 2. tiled matmul
    // --------------------------------------------------------------------------------

    checkCudaErrors(cudaEventRecord(start));
    for (size_t i = 0; i < nIter; ++i)
    {
        // 模板函数调用得加一层括号避免逗号被识别成宏参数
        checkCudaErrors((TiledMatMulKernel1<M, K, N, blockSizeX, blockSizeY><<<dimGrid, dimBlock>>>(d_A, d_B, d_C)));
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("TiledMatMulKernel1 time cost %f\n", msecTotal);

    checkCudaErrors(cudaMemcpy(h_C2, d_C, sizeC, cudaMemcpyDeviceToHost));

    if (CheckEqual(h_C0, h_C1, sizeC))
    {
        printf("h_C0 == h_C1\n");
    }
    else
    {
        printf("h_C0 != h_C1\n");
    }

    if (CheckEqual(h_C1, h_C2, sizeC))
    {
        printf("h_C1 == h_C2\n");
    }
    else
    {
        printf("h_C1 != h_C2\n");
    }

    /*
    printf("Matrix A:\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            printf("%f ", h_A[i * K + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", h_B[i * N + j]);
        }
        printf("\n");
    }
    */

    printf("\nMatrix C0 (Result):\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", h_C0[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C1 (Result):\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", h_C1[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C2 (Result):\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", h_C2[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);

    return 0;
}

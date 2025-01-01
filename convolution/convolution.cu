#include <iostream>
#include <fstream>
#include <memory>
#include <cuda_runtime.h>

// Function to read data from binary file into a unique_ptr
std::unique_ptr<float[]> readDataBin(const std::string &filename, size_t &data_size)
{
    // Open the file in binary mode and move the pointer to the end to get the file size
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr; // Return null if file cannot be opened
    }

    // Calculate the number of elements (assuming float) based on file size
    data_size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg); // Move the file pointer back to the beginning

    // Create a unique_ptr to manage the memory for the data
    std::unique_ptr<float[]> data(new float[data_size]);

    // Read the binary data into the array
    file.read(reinterpret_cast<char *>(data.get()), data_size * sizeof(float));

    if (file)
    {
        std::cout << "Data read successfully!" << std::endl;
    }
    else
    {
        std::cerr << "Error reading data from file." << std::endl;
        return nullptr; // Return null if reading fails
    }

    return data; // Return the unique_ptr containing the data
}

std::unique_ptr<float[]> conv2dCPU(const std::unique_ptr<float[]> &data,
                                   const std::unique_ptr<float[]> &filter,
                                   const size_t height,
                                   const size_t width,
                                   const int radius)
{
    std::unique_ptr<float[]> output(new float[height * width]);
    for (int rowOut = 0; rowOut < height; ++rowOut)
    {
        for (int colOut = 0; colOut < width; ++colOut)
        {
            float value = 0.0f;
            for (int rowFilter = 0; rowFilter < 2 * radius + 1; ++rowFilter)
            {
                for (int colFilter = 0; colFilter < 2 * radius + 1; ++colFilter)
                {
                    int rowIn = rowOut - radius + rowFilter;
                    int colIn = colOut - radius + colFilter;
                    if (0 <= rowIn && rowIn < height && 0 <= colIn && colIn < width)
                    {
                        value += filter[rowFilter * (2 * radius + 1) + colFilter] * data[rowIn * width + colIn];
                    }
                }
            }
            output[rowOut * width + colOut] = value;
        }
    }
    return output;
}

// 最简单的实现
__global__ void conv2dBasicKernel(float *data, float *filter, float *output, int height, int width, int radius)
{
    int rowOut = blockIdx.y * blockDim.y + threadIdx.y;
    int colOut = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    for (int rowFilter = 0; rowFilter < 2 * radius + 1; ++rowFilter)
    {
        for (int colFilter = 0; colFilter < 2 * radius + 1; ++colFilter)
        {
            int rowIn = rowOut - radius + rowFilter;
            int colIn = colOut - radius + colFilter;
            if (0 <= rowIn && rowIn < height && 0 <= colIn && colIn < width)
            {
                value += filter[rowFilter * (2 * radius + 1) + colFilter] * data[rowIn * width + colIn];
            }
        }
    }
    output[rowOut * width + colOut] = value;
}

// filter 放在 constant memory 上, 其他照搬 basic
#define FILTER_RADIUS 2
__constant__ float FILTER[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];
__global__ void conv2dConstMemFilterKernel(float *data, float *output, int height, int width)
{
    int rowOut = blockIdx.y * blockDim.y + threadIdx.y;
    int colOut = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    for (int rowFilter = 0; rowFilter < 2 * FILTER_RADIUS + 1; ++rowFilter)
    {
        for (int colFilter = 0; colFilter < 2 * FILTER_RADIUS + 1; ++colFilter)
        {
            int rowIn = rowOut - FILTER_RADIUS + rowFilter;
            int colIn = colOut - FILTER_RADIUS + colFilter;
            if (0 <= rowIn && rowIn < height && 0 <= colIn && colIn < width)
            {
                value += FILTER[rowFilter * (2 * FILTER_RADIUS + 1) + colFilter] * data[rowIn * width + colIn];
            }
        }
    }
    output[rowOut * width + colOut] = value;
}

// tiled conv2d, block size 和 input tile 对齐
#define OUT_TILE_DIM 4
#define IN_TILE_DIM ((OUT_TILE_DIM) + 2 * (FILTER_RADIUS))
__global__ void conv2dInputTileAligned(float *data, float *output, int height, int width)
{
    // global 坐标
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;

    __shared__ float inputTile[IN_TILE_DIM * IN_TILE_DIM];
    // 对于 output tile 来说不会跑到边界以外, 对于 input tile 来说扩充了一圈 2r 可能越界
    if (0 <= row && row < height && 0 <= col && col < width)
    {
        inputTile[threadIdx.y * IN_TILE_DIM + threadIdx.x] = data[row * width + col];
    }
    else // input 跑到边界以外了, 所谓的 ghost cells, 但这不是 filter 造成的, 而是 input tile 造成的
    {
        inputTile[threadIdx.y * IN_TILE_DIM + threadIdx.x] = 0;
    }
    __syncthreads();

    int rowTile = threadIdx.y - FILTER_RADIUS;
    int colTile = threadIdx.x - FILTER_RADIUS;

    if (0 <= row && row < height && 0 <= col && col < width)
    {
        if (0 <= rowTile && rowTile < OUT_TILE_DIM && 0 <= colTile && colTile < OUT_TILE_DIM)
        {
            float value = 0.0f;
            for (int rowFilter = 0; rowFilter < 2 * FILTER_RADIUS + 1; ++rowFilter)
            {
                for (int colFilter = 0; colFilter < 2 * FILTER_RADIUS + 1; ++colFilter)
                {
                    value += FILTER[rowFilter * (2 * FILTER_RADIUS + 1) + colFilter] * inputTile[(rowTile + rowFilter) * IN_TILE_DIM + (colTile + colFilter)];
                }
            }
            output[row * width + col] = value;
        }
    }
}

// tiled conv2d, block size 和 output tile 对齐
__global__ void conv2dOutputTileAligned(float *data, float *output, int height, int width)
{
    // TODO
}

int main()
{
    size_t data_size, filter_size;
    std::unique_ptr<float[]> data = readDataBin("convolution/conv2d_data_16x16.bin", data_size);
    std::unique_ptr<float[]> filter = readDataBin("convolution/conv2d_filter_5x5.bin", filter_size);
    int height = 16, width = 16, radius = 2;
    size_t data_size_byte = height * width * sizeof(float), filter_size_byte = (2 * radius + 1) * (2 * radius + 1) * sizeof(float);

    auto printData = [&](float *d) {
        std::cout << "--------------------------------------------------------------------------------\n";
        for (size_t i = 0; i < 16; ++i)
        {
            for (size_t j = 0; j < 16; ++j)
            {
                std::cout << d[i * 16 + j] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "--------------------------------------------------------------------------------\n";
    };

    // --------------------------------------------------------------------------------
    // CPU 版本
    // --------------------------------------------------------------------------------
    auto outputCPU = conv2dCPU(data, filter, height, width, radius);
    std::cout << "conv2dCPU\n";
    printData(outputCPU.get());

    // --------------------------------------------------------------------------------
    // basic kernel
    // --------------------------------------------------------------------------------
    float *data_d, *output_d, *filter_d;
    cudaMalloc((void **)&data_d, data_size_byte);
    cudaMalloc((void **)&output_d, data_size_byte);
    cudaMalloc((void **)&filter_d, filter_size_byte);
    std::unique_ptr<float[]> output_h(new float[height * width]);

    cudaMemcpy(data_d, data.get(), data_size_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d, filter.get(), filter_size_byte, cudaMemcpyHostToDevice);
    dim3 dimGrid(4, 4, 1), dimBlock(4, 4, 1);
    conv2dBasicKernel<<<dimGrid, dimBlock>>>(data_d, filter_d, output_d, height, width, radius);
    cudaMemcpy(output_h.get(), output_d, data_size_byte, cudaMemcpyDeviceToHost);
    std::cout << "conv2dBasicKernel\n";
    printData(output_h.get());

    // --------------------------------------------------------------------------------
    // constant memory kernel
    // --------------------------------------------------------------------------------
    cudaMemcpyToSymbol(FILTER, filter.get(), 25 * sizeof(float));
    conv2dConstMemFilterKernel<<<dimGrid, dimBlock>>>(data_d, output_d, height, width);
    cudaMemcpy(output_h.get(), output_d, data_size_byte, cudaMemcpyDeviceToHost);
    std::cout << "conv2dConstMemFilterKernel\n";
    printData(output_h.get());

    // --------------------------------------------------------------------------------
    // conv2dInputTileAligned
    // --------------------------------------------------------------------------------
    cudaMemcpyToSymbol(FILTER, filter.get(), 25 * sizeof(float));
    dim3 dimGrid2(2, 2, 1), dimBlock2(8, 8, 1);
    conv2dInputTileAligned<<<dimGrid2, dimBlock2>>>(data_d, output_d, height, width);
    cudaMemcpy(output_h.get(), output_d, data_size_byte, cudaMemcpyDeviceToHost);
    std::cout << "conv2dInputTileAligned\n";
    printData(output_h.get());

    return 0;
}

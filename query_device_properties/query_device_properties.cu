#include <cuda_runtime.h>
#include <iostream>

void printDeviceProperties(int deviceId)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);

    if (err != cudaSuccess)
    {
        std::cerr << "Error fetching device properties: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Device " << deviceId << ": " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB\n";
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max threads dimensions: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
    std::cout << "  Max grid size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
    std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000.0 << " MHz\n";
    std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "  L2 cache size: " << prop.l2CacheSize / 1024.0 << " KB\n";
    std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << "\n";
    std::cout << "  Clock rate: " << prop.clockRate / 1000.0 << " MHz\n";
    std::cout << "  Device overlap: " << (prop.deviceOverlap ? "Yes" : "No") << "\n";
    std::cout << std::endl;
}

int main()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess)
    {
        std::cerr << "Error fetching device count: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << "\n";

    for (int i = 0; i < deviceCount; ++i)
    {
        printDeviceProperties(i);
    }

    return 0;
}

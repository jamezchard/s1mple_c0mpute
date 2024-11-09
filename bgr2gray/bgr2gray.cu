#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

// static inline void CheckCall(cudaError err, const char *msg, const char *file_name, const int line_number)
// {
//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name,
//                 line_number, cudaGetErrorString(err));
//         std::cin.get();
//         exit(EXIT_FAILURE);
//     }
// }

// #define CHECK_CALL(call, msg) CheckCall((call), (msg), __FILE__, __LINE__)

__global__ void bgr2grayKernel(uchar *img_in, uchar *img_out, int step_in, int step_out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int offset_in = y * step_in + x * 3;
        int offset_out = y * step_out + x;
        // opencv is bgr in memory
        unsigned char b = img_in[offset_in], g = img_in[offset_in + 1], r = img_in[offset_in + 2];
        img_out[offset_out] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

int main(int argc, char **argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_INFO);
    if (argc != 2)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    namedWindow("input image", WINDOW_AUTOSIZE);
    namedWindow("output image", WINDOW_AUTOSIZE);

    // create mat
    Mat img_in, img_out;
    std::string input_filename = argv[1];
    img_in = imread(input_filename, IMREAD_COLOR);
    img_out = Mat(img_in.rows, img_in.cols, CV_8UC1);

    // allocate device memory
    size_t size_in = img_in.step * img_in.rows, size_out = img_out.step * img_out.rows;
    unsigned char *img_in_d, *img_out_d;
    cudaMalloc((void **)&img_in_d, size_in);
    cudaMalloc((void **)&img_out_d, size_out);

    // copy data from host memory to device memory
    cudaMemcpy(img_in_d, img_in.data, size_in, cudaMemcpyHostToDevice);
    // cudaMemcpy(img_out_d, img_out.data, size_out, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((img_in.cols + dimBlock.x - 1) / dimBlock.x, (img_in.rows + dimBlock.y - 1) / dimBlock.y);
    bgr2grayKernel<<<dimGrid, dimBlock>>>(img_in_d, img_out_d, img_in.step, img_out.step, img_in.cols, img_in.rows);

    // copy back data from device to host memory
    cudaMemcpy(img_out.data, img_out_d, size_out, cudaMemcpyDeviceToHost);

    imshow("input image", img_in);
    imshow("output image", img_out);

    size_t dot_position = input_filename.find_last_of(".");
    std::string output_filename = input_filename.substr(0, dot_position) + "-gray.png";
    cv::imwrite(output_filename, img_out);

    waitKey(0);

    return 0;
}

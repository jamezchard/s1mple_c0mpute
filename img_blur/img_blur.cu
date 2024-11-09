#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

__global__ void imgblurKernel(uchar *img_in, uchar *img_out, int step, int width, int height, int blurSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int num = 0, value = 0;
        for (int i = -blurSize; i <= blurSize; ++i)
        {
            for (int j = -blurSize; j <= blurSize; ++j)
            {
                int x_new = x + j, y_new = y + i;
                if (x_new >= 0 && x_new < width && y_new >= 0 && y_new < height)
                {
                    value += img_in[y_new * step + x_new];
                    num++;
                }
            }
        }
        img_out[y * step + x] = (unsigned char)(value / num);
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
    img_in = imread(input_filename, IMREAD_GRAYSCALE);
    img_out = Mat(img_in.rows, img_in.cols, CV_8UC1);

    // allocate device memory
    size_t sizeByte = img_in.step * img_in.rows;
    unsigned char *img_in_d, *img_out_d;
    cudaMalloc((void **)&img_in_d, sizeByte);
    cudaMalloc((void **)&img_out_d, sizeByte);

    // copy data from host memory to device memory
    cudaMemcpy(img_in_d, img_in.data, sizeByte, cudaMemcpyHostToDevice);

    int delta = 2;
    int blurSize = 1;

    while (true)
    {
        dim3 dimBlock(16, 16);
        dim3 dimGrid((img_in.cols + dimBlock.x - 1) / dimBlock.x, (img_in.rows + dimBlock.y - 1) / dimBlock.y);
        imgblurKernel<<<dimGrid, dimBlock>>>(img_in_d, img_out_d, img_in.step, img_in.cols, img_in.rows, blurSize);
        blurSize += delta;
        if (blurSize >= 41)
        {
            delta = -2;
        }
        else if (blurSize <= 1)
        {
            delta = 2;
        }

        cudaMemcpy(img_out.data, img_out_d, sizeByte, cudaMemcpyDeviceToHost);

        imshow("input image", img_in);
        imshow("output image", img_out);

        auto kk = waitKey(1);
        if (kk == 27)
        {
            break;
        }
    }

    return 0;
}

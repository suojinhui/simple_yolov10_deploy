#include "opencv2/opencv.hpp"
#include "trt_preprocess.h"
#include "utils.h"

namespace preprocess {

void preprocess_resize_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, float* m_inputMemory ) 
{
    cv::Mat tar;
    int resizeW = tar_w;
    int resizeH = tar_h;
    int m_imgArea = tar_h * tar_w;

    cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);

    int index;
    int offset_ch0 = m_imgArea * 0;
    int offset_ch1 = m_imgArea * 1;
    int offset_ch2 = m_imgArea * 2;
    for (int i = 0; i < resizeH; i++) {
        for (int j = 0; j < resizeW; j++) {
            index = i * resizeW * 3 + j * 3;
            m_inputMemory[offset_ch2++] = tar.data[index + 0] / 255.0f;
            m_inputMemory[offset_ch1++] = tar.data[index + 1] / 255.0f;
            m_inputMemory[offset_ch0++] = tar.data[index + 2] / 255.0f;
        }
    }
}

void preprocess_resize_gpu(cv::Mat &h_src, float* d_tar, const int& tar_h, const int& tar_w) 
{
    uint8_t* d_src  = nullptr;

    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;

    int src_size  = height * width * chan * sizeof(uint8_t);

    // 分配device上的src的内存
    CUDA_CHECK(cudaMalloc(&d_src, src_size));

    // 将数据拷贝到device上
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));

    // device上处理resize, BGR2RGB的核函数
    resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_src));
}

} // namespace process
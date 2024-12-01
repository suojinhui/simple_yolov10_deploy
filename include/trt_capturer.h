#ifndef __TRT_CAPTURER_H__
#define __TRT_CAPTURER_H__

#include "iostream"
#include "thread"
#include "linux/videodev2.h"
#include "opencv2/opencv.hpp"
#include "condition_variable"

#define TEGRA_CAMERA_CID_BASE (V4L2_CTRL_CLASS_CAMERA | 0x2000)
#define TEGRA_CAMERA_CID_VI_PREFERRED_STRIDE (TEGRA_CAMERA_CID_BASE + 110)
#define V4L2_BUFFERS_NUM 4

namespace capturer {

double GetCurrentTimestamp();

std::string GetCurrentTimeString();

typedef struct {
    unsigned char *start;
    unsigned int length;
} buffer_cam;


class Capturer
{
private:
    int id_;
    std::string device_;
    cv::Size camera_size_;
    cv::Size output_size_;
    int camera_width_;
    int camera_height_;
    int output_width_;
    int output_height_;

    int fd_;
    unsigned int pixfmt_;
    
    buffer_cam* buffer_cam_ = nullptr;

    int frame_count_;

    bool stop_;

    std::thread capture_thread_;
    std::mutex mutex_;

    int fps_;
    int actual_fps_;
    int flip_method_;

    std::mutex fps_mutex_;

    cv::Mat image_;
    
    std::string label_;
    
public:
    std::mutex image_ready_mutex;
    bool data_ready = false;
    std::condition_variable image_ready_condition_;

    bool initiated_ = false;

public:

    ~Capturer();

    Capturer(int id, int camera_width, int camera_height, int output_width, int output_height, 
                int fps, int flip_method, std::string &label);

    void InitV4L2();

    void RequestCameraBuffer();

    void PrepareBuffers();

    void StartStream();

    void StartCaptureLoop();

    void StartCaptureThread();

    void StopStream();

    void StopCapture();

    void SpyOnFps();

    void Destroy();

    cv::Mat GetImage();
};

} // namespace capturer

#endif // __TRT_CAPTURER_H__
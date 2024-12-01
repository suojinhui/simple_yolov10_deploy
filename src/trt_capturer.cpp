#include "iostream"
#include "thread"
#include "mutex"
#include "condition_variable"
#include "filesystem"

#include "csignal"
#include "fcntl.h"
#include "sys/ioctl.h"
#include "sys/poll.h"
#include "sys/mman.h"
#include "linux/v4l2-common.h"
#include "linux/v4l2-controls.h"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"
#include "libyuv.h"

#include "trt_capturer.h"

namespace capturer {

double GetCurrentTimestamp()
{
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        return 0;
    }
    return ((double)time.tv_sec + (double)time.tv_usec * .000001) * 1000;
}

std::string GetCurrentTimeString()
{
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm timeInfo;
    localtime_r(&time, &timeInfo);

    char time_string[26];
    std::strftime(time_string, sizeof(time_string), "%Y_%m_%d_%H_%M_%S", &timeInfo);
    return time_string;
}


Capturer::Capturer(int id, int camera_width, int camera_height, int output_width, int output_height, int fps, 
                 int flip_method, std::string &label):
                 id_(id), 
                 camera_width_(camera_width), 
                 camera_height_(camera_height), 
                 output_width_(output_width), 
                 output_height_(output_height), 
                 fps_(fps),
                 actual_fps_(0), 
                 flip_method_(flip_method), 
                 device_("/dev/video" + std::to_string(id_)), 
                 pixfmt_(V4L2_PIX_FMT_MJPEG), 
                 stop_(false), 
                 frame_count_(0),
                 label_(label), 
                 image_(output_height_, output_width_, CV_8UC3)
                 {}

Capturer::~Capturer() {}

void Capturer::InitV4L2(){
    // Open device
    fd_ = open(device_.c_str(), O_RDWR);
    if (-1 == fd_)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to open this device.", id_);
        return;
    }

    // Set V4l2 control params
    struct v4l2_control ctrl;
    ctrl.id = TEGRA_CAMERA_CID_VI_PREFERRED_STRIDE;
    ctrl.value = (camera_width_ * 2 + 255) / 256 * 256;
    ioctl(fd_, VIDIOC_S_CTRL, &ctrl);

    // Set Camera output format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = camera_width_;
    fmt.fmt.pix.height = camera_height_;
    fmt.fmt.pix.pixelformat = pixfmt_;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to set output format, camera may not have been released", id_);
        return;
    }

    // Get the real format in case the desired is not supported
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to get output format.", id_);
        return;
    }

    if (fmt.fmt.pix.width != camera_width_ || fmt.fmt.pix.height != camera_height_ || fmt.fmt.pix.pixelformat != pixfmt_)
    {
        camera_width_ = fmt.fmt.pix.width;
        camera_height_ = fmt.fmt.pix.height;
        pixfmt_ = fmt.fmt.pix.pixelformat;
        SPDLOG_WARN("Capturer_{}: Camera may not support desired output format, use {}x{} {} instead.", 
                            id_, camera_width_, camera_height_, pixfmt_);
    }

    // Get video stream params
    struct v4l2_streamparm streamparm;
    memset(&streamparm, 0x00, sizeof(struct v4l2_streamparm));

    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_G_PARM, &streamparm);

    SPDLOG_INFO("Capturer_{}: Succeed in initializing v4l2 components.", id_);
    initiated_ = true;
    return;
}

void Capturer::RequestCameraBuffer(){

    struct v4l2_requestbuffers request_buffers;
    memset(&request_buffers, 0, sizeof(request_buffers));
    request_buffers.count = V4L2_BUFFERS_NUM;
    request_buffers.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request_buffers.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &request_buffers) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to request v4l2 buffers.", id_);
        Destroy();
        exit(-1);
    }
    if (V4L2_BUFFERS_NUM != request_buffers.count)
    {
        SPDLOG_ERROR("Capturer_{}: V4l2 buffers number is not as desired.", id_);
        Destroy();
        exit(-1);
    }

    for (size_t index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buffer;
        memset(&buffer, 0, sizeof(buffer));
        buffer.index = index;
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.flags = V4L2_BUF_FLAG_TSTAMP_SRC_SOE; // timestamp from device

        // Query v4l2 buffer
        if (ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0)
        {
            SPDLOG_ERROR("Capturer_{}: Failed to query v4l2 buffer.", id_);
            Destroy();
            exit(-1);
        }

        // Mmap v4l2 buffer
        buffer_cam_[index].length = buffer.length;
        buffer_cam_[index].start =
            (unsigned char *) mmap (NULL /* start anywhere */,
                    buffer.length,
                    PROT_READ | PROT_WRITE /* required */,
                    MAP_SHARED /* recommended */,
                    fd_, buffer.m.offset);

        if (MAP_FAILED == buffer_cam_[index].start)
        {
            SPDLOG_ERROR("Capturer_{}: Failed to mmap v4l2 buffer.", id_);
            Destroy();
            exit(-1);
        }

        if (ioctl(fd_, VIDIOC_QBUF, &buffer) < 0)
        {
            SPDLOG_ERROR("Capturer_{}: Failed to queue v4l2 buffer.", id_);
            Destroy();
            exit(-1);
        }

    }

    SPDLOG_INFO("Capturer_{}: Succeed in requesting v4l2 buffers.", id_);
    return;
}

void Capturer::PrepareBuffers()
{
    // Allocate buffer context
    buffer_cam_ = (buffer_cam*) malloc(V4L2_BUFFERS_NUM * sizeof(buffer_cam));
    memset(buffer_cam_, 0, V4L2_BUFFERS_NUM * sizeof(buffer_cam));

    if (NULL == buffer_cam_)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to allocate global nv_buffer context.", id_);
        Destroy();
        exit(-1);
    }

    RequestCameraBuffer();

    SPDLOG_INFO("Capturer_{}: Succeed in preparing stream buffers.", id_);
    return;
}

void Capturer::StartStream()
{
    // Start v4l2 streaming
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to start streaming.", id_);
        Destroy();
        exit(-1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    SPDLOG_INFO("Capturer_{}: Succeed in starting streaming.", id_);
    return;
}

void Capturer::StartCaptureLoop()
{
    SPDLOG_INFO("Capturer_{}: Start capture loop.", id_);
    struct pollfd poll_fds[1];
    memset(&poll_fds[0], 0, sizeof(pollfd));
    poll_fds[0].fd = fd_;
    poll_fds[0].events = POLLIN;

    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;

    uint8_t *yuv420p_buffer = new uint8_t[camera_width_ * camera_height_ * 3 / 2];
    uint8_t *scaled_yuv420p_buffer = new uint8_t[output_width_ * output_height_ * 3 / 2];
    uint8_t *flipped_yuv420p_buffer = new uint8_t[output_width_ * output_height_ * 3 / 2];
    uint8_t *undistorted_yuv420p_buffer = new uint8_t[output_width_ * output_height_ * 3 / 2];

    uint8_t *image_buffer;
    uint8_t *output_yuv420p_buffer;

    double start_time = 0.;
    double stop_time = 0.;
    
    std::thread spy_on_fps_thread = std::thread(&Capturer::SpyOnFps, this);

    while (poll(poll_fds, 1, 5000) > 0 && !stop_)
    {
        if (poll_fds[0].revents & POLLIN)
        {
            start_time = GetCurrentTimestamp();
            // Dequeue camera buffer
            if (ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0)
            {
                SPDLOG_ERROR("Capturer_{}: Failed to dequeue camera buffer.", id_);
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(fps_mutex_);
                ++actual_fps_;
            }
            image_buffer = (uint8_t *)(buffer_cam_[buffer.index].start);

            if (pixfmt_ == V4L2_PIX_FMT_MJPEG){
                libyuv::MJPGToI420(image_buffer, buffer_cam_[buffer.index].length,
                                   yuv420p_buffer, camera_width_,
                                   yuv420p_buffer + camera_width_ * camera_height_, camera_width_ / 2,
                                   yuv420p_buffer + camera_width_ * camera_height_ * 5 / 4, camera_width_ / 2,
                                   camera_width_, camera_height_, camera_width_, camera_height_);

                //cv::imdecode(cv::Mat(buffer_cam_[buffer.index].length, 1, CV_8UC1, buffer_cam_[buffer.index].start), cv::IMREAD_UNCHANGED, &image_);
            } else if (pixfmt_ == V4L2_PIX_FMT_YUYV) {
            // Convert YUYV to YUV420P
                libyuv::YUY2ToI420(image_buffer, camera_width_ * 2,
                               yuv420p_buffer, camera_width_,
                               yuv420p_buffer + camera_width_ * camera_height_, camera_width_ / 2,
                               yuv420p_buffer + camera_width_ * camera_height_ * 5 / 4, camera_width_ / 2,
                               camera_width_, camera_height_);
            } else {
                SPDLOG_ERROR("Capturer_{}: Unsupported pixel format {}.", id_, pixfmt_);
                Destroy();
                exit(-1);
            }

            // Scale YUV420P buffer
            if (camera_width_ != output_width_ || camera_height_ != output_height_)
            {
                libyuv::I420Scale(yuv420p_buffer, camera_width_,
                                  yuv420p_buffer + camera_width_ * camera_height_, camera_width_ / 2,
                                  yuv420p_buffer + camera_width_ * camera_height_ * 5 / 4, camera_width_ / 2,
                                  camera_width_, camera_height_,
                                  scaled_yuv420p_buffer, output_width_,
                                  scaled_yuv420p_buffer + output_width_ * output_height_, output_width_ / 2,
                                  scaled_yuv420p_buffer + output_width_ * output_height_ * 5 / 4, output_width_ / 2,
                                  output_width_, output_height_,
                                  libyuv::kFilterNone);
                output_yuv420p_buffer = scaled_yuv420p_buffer;
            }
            else
            {
                output_yuv420p_buffer = yuv420p_buffer;
            }

            // Horizontal flip
            if (1 == flip_method_)
            {
                libyuv::I420Mirror(output_yuv420p_buffer, output_width_,
                                   output_yuv420p_buffer + output_width_ * output_height_, output_width_ / 2,
                                   output_yuv420p_buffer + output_width_ * output_height_ * 5 / 4, output_width_ / 2,
                                   flipped_yuv420p_buffer, output_width_,
                                   flipped_yuv420p_buffer + output_width_ * output_height_, output_width_ / 2,
                                   flipped_yuv420p_buffer + output_width_ * output_height_ * 5 / 4, output_width_ / 2,
                                   output_width_, output_height_);
                output_yuv420p_buffer = flipped_yuv420p_buffer;
            }

            // YUV420P to BGR
            libyuv::I420ToRGB24(output_yuv420p_buffer,
                                output_width_,                                                  // Input width
                                output_yuv420p_buffer + output_width_ * output_height_,         // Input U
                                output_width_ / 2,                                              // Input U stride
                                output_yuv420p_buffer + output_width_ * output_height_ * 5 / 4, // Input V
                                output_width_ / 2,                                              // Input V stride
                                image_.data,                                                    // Output RGB
                                output_width_ * 3,                                              // Output RGB stride
                                output_width_, output_height_);
            
            {
                std::unique_lock<std::mutex> lock(image_ready_mutex);
                data_ready = true;
                image_ready_condition_.notify_all();
            }
            
            
            // Enqueue camera buff
            if (ioctl(fd_, VIDIOC_QBUF, &buffer))
            {
                SPDLOG_ERROR("Capturer_{}: Failed to queue camera buffer.", id_);
                Destroy();
                exit(-1);
            }

            stop_time = GetCurrentTimestamp();
            SPDLOG_INFO("Capturer_{}: frame {} time = {} ms", id_, frame_count_, stop_time - start_time);
            ++frame_count_;
        }
    }

    if (spy_on_fps_thread.joinable())
    {
        spy_on_fps_thread.join();
    }
    image_buffer = nullptr;
    output_yuv420p_buffer = nullptr;

    uint8_t *release_pointers[] = {yuv420p_buffer, scaled_yuv420p_buffer, flipped_yuv420p_buffer, undistorted_yuv420p_buffer};

    for (uint8_t *&ptr : release_pointers)
    {
        if (ptr)
        {
            delete[] ptr;
            ptr = nullptr;
        }
    }
    return;
}

void Capturer::StartCaptureThread()
{
    InitV4L2();

    if (!initiated_)
    {
        raise(SIGINT);
        return;
    }

    PrepareBuffers();

    StartStream();

    capture_thread_ = std::thread(&Capturer::StartCaptureLoop, this);
    return;
}

void Capturer::StopStream()
{
    // Stop v4l2 streaming
    if (!initiated_)
        return;

    SPDLOG_INFO("Capturer_{}: Stop capture loop.", id_);
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type))
    {
        SPDLOG_ERROR("Capturer_{}: Failed to stop streaming", id_);
        return;
    }

    SPDLOG_INFO("Capturer_{}: Succeed in stopping streaming", id_);
    return;
}

void Capturer::StopCapture()
{
    StopStream();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }

    actual_fps_ = -1;

    if (capture_thread_.joinable())
    {
        capture_thread_.join();
    }

    Destroy();

    SPDLOG_INFO("Capturer_{}: Stopped", id_);

    return;
}

void Capturer::SpyOnFps()
{
    int id;
    while (!stop_)
    {
        id = id_;
        {
            std::lock_guard<std::mutex> lock(fps_mutex_);
            actual_fps_ = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        {
            std::lock_guard<std::mutex> lock(fps_mutex_);
            if (actual_fps_ != -1 && abs(actual_fps_ - fps_) > 1)
            {
                if (stop_)
                {
                    actual_fps_ = -1;
                    return;
                }
                std::stringstream ss;
                ss << "Capturer_" << id << " fps = " << actual_fps_;
                std::cout << ss.str() << std::endl;
                SPDLOG_WARN("Capturer_{}: WRANING! FPS has decreased, FPS = {}", id, actual_fps_);
            }
            actual_fps_ = 0;
        }
    }

    return;
}

void Capturer::Destroy()
{
    actual_fps_ = -1;

    if (fd_ > 0)
    {
        close(fd_);
    }

    if (buffer_cam_ != nullptr)
    {
        for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++)
        {
            auto &buffer = buffer_cam_[i];

            if (buffer.start != nullptr) {
                munmap(buffer.start, buffer.length);
                buffer.start = nullptr;
            }
        }
        free(buffer_cam_);
        buffer_cam_ = nullptr;   
    }
    SPDLOG_INFO("Capturer_{}: Succeed in destroying v4l2 components.", id_);
    return;
}

cv::Mat Capturer::GetImage()
{
    {
        std::unique_lock<std::mutex> lock(image_ready_mutex);
        image_ready_condition_.wait(lock, [this]()
                                    { return data_ready; });
        data_ready = false;
    }
    SPDLOG_INFO("Capturer_{}: GetImage", id_);
    return image_;
}

} // namespace capturer
#ifndef __YOLOV10_TRT_MODEL_H__
#define __YOLOV10_TRT_MODEL_H__

#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include "trt_logger.h"
#include "trt_label.h"

#define WORKSPACESIZE 1<<28

namespace model {

/* device type */
enum device {
    CPU = 0,
    GPU = 1
};

enum precision {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2  // 需要量化
};

// image info
struct image_info {
    int h;
    int w;
    int c;
    image_info(int height, int width, int channel) : h(height), w(width), c(channel) {}
};

struct bbox {
    float x0, x1, y0, y1;
    float confidence;
    bool  flg_remove;
    int   label;
    
    bbox() = default;
    bbox(float x0, float y0, float x1, float y1, float conf, int label) : 
        x0(x0), y0(y0), x1(x1), y1(y1), 
        confidence(conf), flg_remove(false), 
        label(label){};
};

template<typename T>
void destroy_trt_ptr(T* ptr){
    if (ptr) {
        std::string type_name = typeid(T).name();
        LOGD("Destroy %s", type_name.c_str());
        delete ptr; 
    };
}

struct InferDeleter {
    template<typename T>
    void operator()(T* obj) const {
        if (obj) {
            destroy_trt_ptr<T>(obj);
        }
    }
};

class YOLOv10TRT {
public:
    // constructor and destructor
    YOLOv10TRT(std::string engine_path, std::string onnx_path, std::string output_path, logger::Level level,
            device device_type, precision precision_type, int num_classes, image_info img_info, float conf_threshold, 
            float nms_threshold, label::Datasets dataset_type);
    ~YOLOv10TRT();

    // runtime functions
    void load_image(cv::Mat& img);
    void inference(cv::Mat& img);
    bool enqueue_bindings();
    void reset_task();
    bool preprocess_cpu();
    bool postprocess_cpu();
    bool preprocess_gpu();
    bool postprocess_gpu();

    // building functions
    void init_model();
    bool load_engine();
    bool build_engine();
    void setup(void* data, std::size_t size);
    void save_plan(nvinfer1::IHostMemory& plan);

public:
    std::string m_enginePath;
    std::string m_onnxPath;
    std::string m_outputPath;

    cv::Mat m_inputImage;

    device m_device;
    precision m_precision;
    int m_numclasses;
    image_info m_imginfo;
    int m_workspacesize = WORKSPACESIZE;

    float* m_bindings[2];
    float* m_inputMemory[2];
    float* m_outputMemory[2];
    int m_inputSize;
    int m_outputSize;

    float m_conf_threshold = 0.3;
    float m_nms_threshold  = 0.45;
    std::shared_ptr<label::IBaseLabels> labels_map;

    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;
    cudaStream_t m_stream;

    std::shared_ptr<logger::Logger> m_logger;
    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    std::shared_ptr<nvinfer1::INetworkDefinition> m_network;

    std::vector<bbox> m_bboxes;
};

std::shared_ptr<YOLOv10TRT> make_detector(std::string engine_path, std::string onnx_path, std::string output_path, logger::Level level,
            device device_type, precision precision_type, int num_classes, image_info img_info, float conf_threshold, 
            float nms_threshold, label::Datasets dataset_type);

} // namespace model

#endif // __YOLOV10_TRT_MODEL_H__
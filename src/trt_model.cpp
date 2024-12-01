#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>


#include "trt_logger.h"
#include "trt_model.h"
#include "utils.h"
#include "trt_preprocess.h"
#include "trt_label.h"

using namespace nvinfer1;
using namespace std;
using namespace nvonnxparser;

namespace model{

    YOLOv10TRT::YOLOv10TRT(std::string engine_path, std::string onnx_path, std::string output_path, logger::Level level,
            device device_type, precision precision_type, int num_classes, image_info img_info, float conf_threshold, 
            float nms_threshold, label::Datasets dataset_type): 
            m_enginePath(engine_path), 
            m_onnxPath(onnx_path), 
            m_outputPath(output_path), 
            m_device(device_type), 
            m_precision(precision_type), 
            m_numclasses(num_classes), 
            m_imginfo(img_info),
            m_conf_threshold(conf_threshold),
            m_nms_threshold(nms_threshold) 
            {
                m_logger = logger::create_logger(level);
                if(dataset_type == label::Datasets::COCO)
                {
                    labels_map = shared_ptr<label::IBaseLabels>(new label::CocoLabels());
                } else if (dataset_type == label::Datasets::Bdd100k)
                {
                    labels_map = shared_ptr<label::IBaseLabels>(new label::Bdd100kLabels());
                } else {
                    LOGE("Unsupported dataset type: %d, dataset_type, file: %s, line: %d", __FILE__, __LINE__);
                    std::abort();
                }
            }

    YOLOv10TRT::~YOLOv10TRT() {}

    void YOLOv10TRT::load_image(cv::Mat& img){
        m_inputImage = img;
    }

    void YOLOv10TRT::init_model() {

        if (m_context == nullptr) {
            if (!fileExists(m_enginePath)){
                LOGV("%s not found, building engine...", m_enginePath.c_str());
                if(!build_engine()){
                    LOGE("Failed to build engine, program terminated, file: %s, line: %d", __FILE__, __LINE__);
                    std::abort();
                }
            } else {
                LOGV("Loading engine from %s", m_enginePath.c_str());
                if (!load_engine()){
                    LOGE("Failed to load engine, program terminated, file: %s, line: %d", __FILE__, __LINE__);
                    std::abort();
                }
            }
        } else {
            reset_task();
        }
    }

    bool YOLOv10TRT::build_engine() {

        auto builder = shared_ptr<IBuilder>(createInferBuilder(*m_logger), InferDeleter());
        auto network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(1), InferDeleter());
        auto config  = shared_ptr<IBuilderConfig>(builder->createBuilderConfig(), InferDeleter());
        auto parser  = shared_ptr<IParser>(createParser(*network, *m_logger), InferDeleter());

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, m_workspacesize);
        config->setProfilingVerbosity(ProfilingVerbosity::kLAYER_NAMES_ONLY);

        if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
            LOGE("Failed to parse ONNX file: %s, file: %s, line: %d", m_onnxPath.c_str(), __FILE__, __LINE__);
            return false;
        }

        if (builder->platformHasFastFp16() && m_precision == precision::FP16) {
            config->setFlag(BuilderFlag::kFP16);
            config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        } else if (builder->platformHasFastInt8() && m_precision == precision::INT8) {
            // config->setFlag(BuilderFlag::kINT8);
            // config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            LOGW("INT8 precision is not supported by this platform, using FP16 instead");
            config->setFlag(BuilderFlag::kFP16);
            config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        }

        auto plan = builder->buildSerializedNetwork(*network, *config);
        save_plan(*plan);
        setup(plan->data(), plan->size());

        return true;
    }

    bool YOLOv10TRT::load_engine() {

        vector<unsigned char> modelData;
        ifstream in(m_enginePath, ios::in | ios::binary);
        if (!in.is_open())
            return false;

        in.seekg(0, ios::end);
        size_t length = in.tellg();

        vector<uint8_t> data;
        if (length > 0){
            in.seekg(0, ios::beg);
            data.resize(length);
            in.read((char*)&data[0], length);
        }
        in.close();

        modelData = data;
        setup(modelData.data(), modelData.size());

        return true;
    }

    void YOLOv10TRT::save_plan(IHostMemory& plan) {
        auto f = fopen(m_enginePath.c_str(), "wb");
        fwrite(plan.data(), 1, plan.size(), f);
        fclose(f);
    }

    void YOLOv10TRT::setup(void* data, size_t size) {
        m_runtime    = shared_ptr<IRuntime>(createInferRuntime(*m_logger), InferDeleter());
        m_engine     = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), InferDeleter());
        m_context    = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), InferDeleter());
        m_inputDims = m_engine->getTensorShape(m_engine->getIOTensorName(0));
        m_outputDims = m_engine->getTensorShape(m_engine->getIOTensorName(1));

        CUDA_CHECK(cudaStreamCreate(&m_stream));
        
        if(m_imginfo.w != m_inputDims.d[3] || m_imginfo.h != m_inputDims.d[2]){
            LOGE("Input image size (%d, %d) does not match network input size (%d, %d), file: %s, line: %d", 
                m_imginfo.w, m_imginfo.h, m_inputDims.d[3], m_inputDims.d[2], __FILE__, __LINE__);
                std::abort();
        }

        m_inputSize =  m_imginfo.h * m_imginfo.w * m_imginfo.c * sizeof(float);
        m_outputSize = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);

        CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
        CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));

        CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
        CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));

        m_bindings[0] = m_inputMemory[1];
        m_bindings[1] = m_outputMemory[1];

        m_context->setTensorAddress(m_engine->getIOTensorName(0), m_bindings[0]);
        m_context->setTensorAddress(m_engine->getIOTensorName(1), m_bindings[1]);

        LOGV("Engine loaded successfully");
    }

    void YOLOv10TRT::reset_task() {
        m_bboxes.clear();
    }

    bool YOLOv10TRT::enqueue_bindings() {
        
        // if (!m_context->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        //     LOGE("Error happens during DNN inference part, program terminated, file: %s, line: %d", __FILE__, __LINE__);
        //     return false;
        // }
        if (!m_context->enqueueV3(m_stream)){
            LOGE("Error happens during DNN inference part, program terminated");
            return false;
        }
        return true;
    }

    void YOLOv10TRT::inference(cv::Mat& img) {

        load_image(img);

        if (m_device == device::GPU){
            preprocess_gpu();
        } else {
            preprocess_cpu();
        }

        enqueue_bindings();

        if (m_device == device::GPU){
            postprocess_gpu();
        } else {
            postprocess_cpu();
        }
    }

    bool YOLOv10TRT::preprocess_cpu() {
        if (m_inputImage.data == nullptr){
            LOGE("Image data is empty, please load image first!, file: %s, line: %d", __FILE__, __LINE__);
            return false;
        }

        preprocess::preprocess_resize_cpu(m_inputImage, m_imginfo.h, m_imginfo.w, m_inputMemory[0]);
        CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, 
            cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

        return true;
    }

    bool YOLOv10TRT::postprocess_cpu() {

        reset_task();

        CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], m_outputSize, 
            cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
        CUDA_CHECK(cudaStreamSynchronize(m_stream));

        int boxes_cout = m_outputDims.d[1];
        float* tensor;
        float x1, y1, x0, y0, conf;
        int label;

        for (int i = 0; i < boxes_cout; i++){
            tensor = m_outputMemory[0] + i * m_outputDims.d[2];
            conf  = tensor[4];
            label = tensor[5];
            // LOGD("conf: %f, label: %d", conf, label);

            if (conf < m_conf_threshold) break;

            x0 = tensor[0];
            y0 = tensor[1];
            x1 = tensor[2];
            y1 = tensor[3];

            x0 *= m_inputImage.cols / float(m_imginfo.w);
            y0 *= m_inputImage.rows / float(m_imginfo.h);
            x1 *= m_inputImage.cols / float(m_imginfo.w); 
            y1 *= m_inputImage.rows / float(m_imginfo.h);

            bbox yolov10_box(x0, y0, x1, y1, conf, label);
            m_bboxes.push_back(yolov10_box);
        }
        LOGD("Number of detected objects: %d", m_bboxes.size());

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        float font_scale = 0.001 * std::min(m_inputImage.cols, m_inputImage.rows);
        int font_thickness = 1;
        int baseline = 0;
        std::map<std::string, int> object_count;
        std::ostringstream oss;

        LOG("\tResult:");
        for (auto box : m_bboxes){
            auto name      = labels_map->get_label(box.label);
            auto rec_color = labels_map->get_color(box.label);
            auto txt_color = labels_map->get_inverse_color(rec_color);

            object_count[name]++;

            auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), box.confidence * 100);
            auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thickness, &baseline);

            int txt_height = txt_size.height + baseline + 10;
            int txt_width  = txt_size.width + 3;

            cv::Point txt_pos(round(box.x0), round(box.y0 - (txt_size.height - baseline + font_thickness)));
            cv::Rect  txt_rec(round(box.x0 - font_thickness), round(box.y0 - txt_height), txt_width, txt_height);
            cv::Rect  box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0), round(box.y1 - box.y0));

            cv::rectangle(m_inputImage, box_rec, rec_color, 3);
            cv::rectangle(m_inputImage, txt_rec, rec_color, -1);
            cv::putText(m_inputImage, txt, txt_pos, font_face, font_scale, txt_color, font_thickness, 16);

            LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), (x1, y1)(%6.2f, %6.2f)", 
                name.c_str(), box.confidence * 100, box.x0, box.y0, box.x1, box.y1);

        }
        for (const auto& entry : object_count) {
            oss << entry.first << ": " << entry.second << "  "; // 用空格分隔每个类别及其数量
        }
        LOG("\tSummary:");
        LOG("\t\t%s", oss.str().c_str());
        LOG("\tDetected Objects: %d", m_bboxes.size());

        return true;
}

    bool YOLOv10TRT::preprocess_gpu() {
        if (m_inputImage.data == nullptr){
            LOGE("Image data is empty, please load image first!, file: %s, line: %d", __FILE__, __LINE__);
            return false;
        }

        preprocess::preprocess_resize_gpu(m_inputImage, m_inputMemory[1], m_imginfo.h, m_imginfo.w);

        return true;

    }

    bool YOLOv10TRT::postprocess_gpu() {

        reset_task();

        CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], m_outputSize, 
            cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
        CUDA_CHECK(cudaStreamSynchronize(m_stream));

        int boxes_cout = m_outputDims.d[1];
        float* tensor;
        float x1, y1, x0, y0, conf;
        int label;

        for (int i = 0; i < boxes_cout; i++){
            tensor = m_outputMemory[0] + i*m_outputDims.d[2];
            conf  = tensor[4];
            label = tensor[5];
            // LOGD("conf: %f, label: %d", conf, label);

            if (conf < m_conf_threshold) break;

            x0 = tensor[0];
            y0 = tensor[1];
            x1 = tensor[2];
            y1 = tensor[3];

            preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
            preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);

            bbox yolov10_box(x0, y0, x1, y1, conf, label);
            m_bboxes.push_back(yolov10_box);
        }
        LOGD("Number of detected objects: %d", m_bboxes.size());

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        float font_scale = 0.001 * std::min(m_inputImage.cols, m_inputImage.rows);
        int font_thickness = 1;
        int baseline = 0;
        std::map<std::string, int> object_count;
        std::ostringstream oss;

        LOG("\tResult:");
        for (auto box : m_bboxes){
            auto name      = labels_map->get_label(box.label);
            auto rec_color = labels_map->get_color(box.label);
            auto txt_color = labels_map->get_inverse_color(rec_color);

            object_count[name]++;

            auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), box.confidence * 100);
            auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thickness, &baseline);

            int txt_height = txt_size.height + baseline + 10;
            int txt_width  = txt_size.width + 3;

            cv::Point txt_pos(round(box.x0), round(box.y0 - (txt_size.height - baseline + font_thickness)));
            cv::Rect  txt_rec(round(box.x0 - font_thickness), round(box.y0 - txt_height), txt_width, txt_height);
            cv::Rect  box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0), round(box.y1 - box.y0));

            cv::rectangle(m_inputImage, box_rec, rec_color, 3);
            cv::rectangle(m_inputImage, txt_rec, rec_color, -1);
            cv::putText(m_inputImage, txt, txt_pos, font_face, font_scale, txt_color, font_thickness, 16);

            LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), (x1, y1)(%6.2f, %6.2f)", 
                name.c_str(), box.confidence * 100, box.x0, box.y0, box.x1, box.y1);

        }
        for (const auto& entry : object_count) {
            oss << entry.first << ": " << entry.second << "  ";
        }
        LOG("\tSummary:");
        LOG("\t\t%s", oss.str().c_str());
        LOG("\tDetected Objects: %d", m_bboxes.size());

        return true;
    }

    shared_ptr<YOLOv10TRT> make_detector(std::string engine_path, std::string onnx_path, std::string output_path, logger::Level level,
            device device_type, precision precision_type, int num_classes, image_info img_info, float conf_threshold, 
            float nms_threshold, label::Datasets dataset_type){
        return make_shared<YOLOv10TRT>(engine_path, onnx_path, output_path, level, device_type, precision_type, 
            num_classes, img_info, conf_threshold, nms_threshold, dataset_type);
    }  
    
} // namespace model

# include <iostream>
# include <opencv2/opencv.hpp>

#include "trt_model.h"
#include "trt_logger.h"
#include "utils.h"
#include "trt_label.h"
#include "trt_capturer.h"

bool quit = false;

int main(int argc, char* argv[]) {

    // Read config file
    std::string onnx_path;
    std::string engine_path;
    std::string output_path;
    logger::Level level ; int log_level;
    model::device device_type ; int model_device_type;
    model::precision precision_type; int model_precision_type;
    int witdh, height;
    int num_classes;
    float conf_threshold, nms_threshold;
    label::Datasets dataset_type; int label_dataset_type;

    cv::FileStorage config_file("./config/config.yaml", cv::FileStorage::READ);
    if (!config_file.isOpened()) {
        LOGE("Error opening config file, file: %s, line: %d", __FILE__, __LINE__);
        return -1;
    }
    config_file["onnx_path"] >> onnx_path;
    config_file["engine_path"] >> engine_path;
    config_file["output_path"] >> output_path;
    config_file["log_level"] >> log_level;
    config_file["device_type"] >> model_device_type;
    config_file["precision_type"] >> model_precision_type;
    config_file["width"] >> witdh;
    config_file["height"] >> height;
    config_file["num_classes"] >> num_classes;
    config_file["conf_threshold"] >> conf_threshold;
    config_file["nms_threshold"] >> nms_threshold;
    config_file["dataset_type"] >> label_dataset_type;

    level = static_cast<logger::Level>(log_level);
    device_type = static_cast<model::device>(model_device_type);
    precision_type = static_cast<model::precision>(model_precision_type);
    dataset_type = static_cast<label::Datasets>(label_dataset_type);

    // print config
    LOG("*********************CONFIG FILE***********************");
    LOG("\tONNX path:            %s", onnx_path.c_str());
    LOG("\tEngine path:          %s", engine_path.c_str());
    LOG("\tOutput path:          %s", output_path.c_str());
    LOG("\tLog level:            %d", log_level);
    LOG("\tDevice type:          %d", model_device_type);
    LOG("\tPrecision type:       %d", model_precision_type);
    LOG("\tImage width:          %d", witdh);
    LOG("\tImage height:         %d", height);
    LOG("\tNumber of classes:    %d", num_classes);
    LOG("\tConfidence threshold: %f", conf_threshold);
    LOG("\tNMS threshold:        %f", nms_threshold);
    LOG("\tDataset type:         %d", label_dataset_type);
    LOG("*************************END***************************");

    // Create model
    auto YOLOV10_trt_model = model::make_detector(engine_path, onnx_path, output_path, level, device_type,
            precision_type, num_classes, model::image_info(height,witdh,3), conf_threshold, nms_threshold, dataset_type);

    // setup model
    YOLOV10_trt_model->init_model();

    // // Read video
    // cv::VideoCapture cap("./data/video_test.mp4");
    // // cv::VideoCapture cap(0);

    // cv::Mat frame;
    // if (!cap.isOpened()) {
    //     LOGE("Error opening video stream or file, file: %s, line: %d", __FILE__, __LINE__);
    //     return -1;
    // } else {
    //     cap >> frame;
    //     LOGV("Video stream opened successfully");
    //     LOGV("video rate: %d, size: %d x %d", cap.get(cv::CAP_PROP_FPS), frame.cols, frame.rows);
    // }

    // // Start inference

    // char key;

    // while (cap.isOpened()) {
    //     auto start = chrono::high_resolution_clock::now();

    //     cap >> frame;
    //     if (frame.empty()) break;

    //     YOLOV10_trt_model->inference(frame);
    //     cv::imshow("detection result", frame);

    //     auto end = chrono::high_resolution_clock::now();
    //     auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    //     LOGV("Inference time: %f ms", duration.count()*0.001);

    //     if(34-duration.count()*0.001<=0){
    //         cv::waitKey(1);
    //     }else{
    //         cv::waitKey(int(34-duration.count()*0.001)+1);
    //     }
    //     key = cv::waitKey(1);
    //     if (key == 'Q' || key == 'q')
    //     {
    //         break;
    //     }

    // }
    // cap.release();
    // cv::destroyAllWindows();

    std::string label = "v42l_camera";
    capturer::Capturer my_capturer(0, 1280, 720, 1280, 720, 30, 0, label);
    my_capturer.StartCaptureThread();

    char key;
    int cnt = 0;

    while(true){

        cv::Mat frame = my_capturer.GetImage();
        // 推理
        auto start = chrono::high_resolution_clock::now();
        YOLOV10_trt_model->inference(frame);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        LOGV("Inference time: %f ms", duration.count()*0.001);

        cv::namedWindow("v42l_camera", cv::WINDOW_NORMAL);
        cv::imshow("v42l_camera", frame);

        key = cv::waitKey(1);
        if (key == 'Q' || key == 'q')
        {
            break;
        }
    }

    while (!quit)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        quit = true;
    }

    my_capturer.StopCapture();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    return 0;
}



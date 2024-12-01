# yolov10_v4l2_tensorrt_deploy

## 介绍

yolov10_v4l2_tensorrt_deploy基于v4l2_camera和tensorrt实现了yolov10目标检测，并通过opencv输出到屏幕。
![效果图](/docs/image1.png)

推理速度3ms/帧左右，完全满足实时性要求。

## 依赖

* opencv
* v4l2
* spdlog
* libyuv
* tensorrt10.0以上版本
* cuda12.4

## 用法

* 配置环境变量、库的路径

* 训练模型，并将模型文件导出为`.onnx`格式文件，放到`model/onnx`目录下。

* 确定摄像头的参数，可以使用v4l2-ctl命令：

* 修改`config/config.yaml`文件

* 编译：

```
make
```

* 运行：

```
./bin/trt-yolov10-app
```
* 按下`q`键退出程序。

## 注意事项

* 目前仅支持Linux系统。
* int8量化未实现。
* 目前仅支持单路摄像头。
* 自定数据集需要自行实现`IBaseLabels`的派生类，并在对应枚举`Datasets`中添加相应的标签。

## TODO

* 支持多路摄像头。
* 支持int8量化。
* 优化传参逻辑。
* 编写cuda kernel来支持图像格式转化，提高取图效率。
* 添加支持tensorrt8.0的实现分支。

## 联系方式

如果有任何问题，欢迎联系我：<suojinhui@gmail.com>
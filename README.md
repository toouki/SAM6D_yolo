
---

# 详细使用文档

## 项目概述

SAM-6D 是一个基于 Segment Anything Model 的零样本 6D 物体姿态估计框架。该项目能够从 RGB-D 图像中检测并估计物体的 6D 姿态（3D 位置 + 3D 旋转），无需针对特定物体进行训练。

可使用Utralytics的无监督分割网络进行零样本检测，或全监督网络对物体进行训练。

## 核心功能

### 1. 实例分割模型 (Instance Segmentation Model)
- **功能**: 从 RGB 图像中检测并分割出目标物体
- **支持模型**: SAM (Segment Anything Model)、FastSAM、DINOv2
- **输入**: RGB 图像 + CAD 模型
- **输出**: 物体掩码和边界框

### 2. 姿态估计模型 (Pose Estimation Model)
- **功能**: 估计物体的 6D 姿态（位置 + 旋转）
- **技术**: 基于 PointNet++ 的深度学习架构
- **输入**: RGB-D 图像 + 分割结果 + CAD 模型
- **输出**: 6D 姿态参数

### 3. 模板渲染 (Template Rendering)
- **功能**: 为 CAD 模型生成多视角模板图像
- **工具**: BlenderProc
- **用途**: 用于实例分割模型的模板匹配

## 安装指南

### 系统要求（测试可用）
- Python 3.13
- CUDA 12.8
- Blender

### 环境配置

1. **克隆项目**
```bash
git clone https://github.com/JiehongLin/SAM-6D.git
cd SAM-6D
```

2. **创建 Conda 环境**
```bash
conda env create -f environment.yaml
conda activate sam6d
or
pip install -r requirements.txt
```

3. **安装 PointNet2**
```bash
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
cd ../../../
```

4. **下载预训练模型**
```bash
# 下载实例分割模型
cd Instance_Segmentation_Model
python download_sam.py      # SAM 模型
python download_fastsam.py  # FastSAM 模型
python download_dinov2.py   # DINOv2 模型
cd ../

# 下载姿态估计模型
cd Pose_Estimation_Model
python download_sam6d-pem.py
cd ../
```

## 使用方法

### 快速开始

1. **准备输入数据**
   - CAD 模型文件（.ply 格式，单位：毫米）
   - RGB 图像（.png/.jpg 格式）
   - 深度图像（.png 格式，单位：毫米）
   - 相机内参文件（.json 格式）

2. **渲染模版**
```bash
blenderproc run ./Render/render_custom_templates.py \
--output_dir Data/Example/outputs \
--cad_path Data/Example/obj_000005.ply
```

3. **实例分割**
```bash
python Instance_Segmentation_Model/run_inference_custom.py \
--segmentor_model fastsam \
--output_dir Data/Example/outputs \
--cad_path Data/Example/obj_000005.ply \
--rgb_path Data/Example/rgb.png \
--depth_path Data/Example/depth.png \
--cam_path Data/Example/camera.json
```
4. **位姿估计**
```bash
python Pose_Estimation_Model/run_inference_custom.py \
--output_dir Data/Example/outputs \
--cad_path Data/Example/obj_000005.ply \
--rgb_path Data/Example/rgb.png \
--depth_path Data/Example/depth.png \
--cam_path Data/Example/camera.json \
--seg_path Data/Example/outputs/sam6d_results/detection_ism.json \
```

## 输入数据格式

### CAD 模型
- **格式**: PLY (Polygon File Format)
- **单位**: 毫米 (mm)
- **要求**: 三角网格模型，包含顶点和面信息

### RGB 图像
- **格式**: PNG、JPG
- **要求**: 标准 RGB 彩色图像

### 深度图像
- **格式**: PNG（16位无符号整数）
- **单位**: 毫米 (mm)
- **要求**: 与 RGB 图像对齐

### 相机内参文件
```json
{
    "fx": 525.0,  // 焦距 x
    "fy": 525.0,  // 焦距 y
    "cx": 319.5,  // 主点 x
    "cy": 239.5,  // 主点 y
    "width": 640, // 图像宽度
    "height": 480 // 图像高度
}
```

## 输出结果

### 结果文件结构
```
OUTPUT_DIR/
├── sam6d_results/
│   ├── detection_ism.json          # 实例分割结果
│   ├── pose_estimation_results.json # 姿态估计结果
│   ├── templates/                  # 渲染的模板图像
│   └── visualizations/             # 可视化结果
├── obj_000005_templates/           # CAD 模型模板
└── logs/                           # 运行日志
```

### 姿态估计结果格式
```json
{
    "poses": [
        {
            "obj_id": "obj_000005",
            "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],  // 旋转矩阵
            "t": [tx, ty, tz],  // 平移向量（毫米）
            "score": 0.95       // 置信度分数
        }
    ]
}
```

---  

# <p align="center"> <font color=#008000>SAM-6D</font>: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation </p>

####  <p align="center"> [Jiehong Lin](https://jiehonglin.github.io/), [Lihua Liu](https://github.com/foollh), [Dekun Lu](https://github.com/WuTanKun), [Kui Jia](http://kuijia.site/)</p>
#### <p align="center">CVPR 2024 </p>
#### <p align="center">[[Paper]](https://arxiv.org/abs/2311.15707) </p>

<p align="center">
  <img width="100%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/vis.gif"/>
</p>


## News
- [2024/03/07] We publish an updated version of our paper on [ArXiv](https://arxiv.org/abs/2311.15707).
- [2024/02/29] Our paper is accepted by CVPR2024!


## Update Log
- [2024/03/05] We update the demo to support [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), you can do this by specifying `SEGMENTOR_MODEL=fastsam` in demo.sh.
- [2024/03/03] We upload a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for running custom data.
- [2024/03/01] We update the released [model](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing) of PEM. For the new model, a larger batchsize of 32 is set, while that of the old is 12. 

## Overview
In this work, we employ Segment Anything Model as an advanced starting point for **zero-shot 6D object pose estimation** from RGB-D images, and propose a novel framework, named **SAM-6D**, which utilizes the following two dedicated sub-networks to realize the focused task:
- [x] [Instance Segmentation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)
- [x] [Pose Estimation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Pose_Estimation_Model)


<p align="center">
  <img width="50%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_sam_6d.png"/>
</p>


## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/JiehongLin/SAM-6D.git
```
Install the environment and download the model checkpoints:
```
cd SAM-6D
sh prepare.sh
```
We also provide a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for convenience.

### 2. Evaluation on the custom data
```
# set the paths
export CAD_PATH=Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=Data/Example/outputs         # path to a pre-defined file for saving results

# run inference
cd SAM-6D
sh demo.sh
```



## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Contact

If you have any questions, please feel free to contact the authors. 

Jiehong Lin: [mortimer.jh.lin@gmail.com](mailto:mortimer.jh.lin@gmail.com)

Lihua Liu: [lihualiu.scut@gmail.com](mailto:lihualiu.scut@gmail.com)

Dekun Lu: [derkunlu@gmail.com](mailto:derkunlu@gmail.com)

Kui Jia:  [kuijia@gmail.com](kuijia@gmail.com)

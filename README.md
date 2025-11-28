
---

# 详细使用文档

## 项目概述

SAM-6D 是一个基于 Segment Anything Model 的零样本 6D 物体姿态估计框架。该项目能够从 RGB-D 图像中检测并估计物体的 6D 姿态（3D 位置 + 3D 旋转），无需针对特定物体进行训练。

可使用Utralytics的无监督分割网络进行零样本检测（优化至2s以内），或全监督网络对物体进行训练（2FPS左右）。

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
- CUDA 12.9
- Pytorch 2.9
- Blender 5.0

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
cd SAM-6D
blenderproc run ./Render/render_custom_templates.py \
--output_dir Data/Example/outputs \
--cad_path Data/Example/obj_000005.ply
```

3. **实例分割**
```bash
cd SAM-6D
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
cd SAM-6D
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
├── detection_ism.json              # 实例分割结果
├── pose_estimation_results.json    # 姿态估计结果
├── templates/                      # 渲染的模板图像
├── vis_segmentation.png...         # 实例分割可视化结果
├── vis_pose.png...                 # 姿态估计可视化结果
└── descriptors/                    # CAD 特征描述符
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

## 配置文件参数说明

### 实例分割模型配置 (ISM)

#### 主配置文件参数 (ISM_*.yaml)

| 参数名 | 类型 | 默认值 |  说明  |
|--------|------|--------|------|
| `_target_` | str | model.detector.Instance_Segmentation_Model | 目标检测器类路径 |
| `log_interval` | int | 5 | 日志输出间隔（迭代次数） |
| `log_dir` | str | ${save_dir} | 日志保存目录 |
| `segmentor_width_size` | int | 640 | 分割器输入图像宽度（保持稳定性） |
| `descriptor_width_size` | int | 640 | 描述符网络输入图像宽度 |
| `visible_thred` | float | 0.5 | 可见性阈值，用于过滤遮挡严重的物体 |
| `pointcloud_sample_num` | int | 2048 | 点云采样数量 |

#### 后处理配置 (post_processing_config)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `mask_post_processing.min_box_size` | float | 0.05 | 最小边界框尺寸（相对于图像尺寸） |
| `mask_post_processing.min_mask_size` | float | 3e-4 | 最小掩码尺寸（相对于图像尺寸） |
| `nms_thresh` | float | 0.25 | 非极大值抑制阈值 |

#### 匹配配置 (matching_config)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `metric._target_` | str | model.loss.PairwiseSimilarity | 相似度计算类路径 |
| `metric.metric` | str | cosine | 相似度度量类型（余弦相似度） |
| `metric.chunk_size` | int | 16 | 分块处理大小，用于内存优化 |
| `aggregation_function` | str | avg_5 | 聚合函数类型（平均5个最佳匹配） |
| `confidence_thresh` | float | 0.2 | 置信度阈值，用于过滤低置信度匹配 |

#### 初始化配置 (onboarding_config)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `rendering_type` | str | pbr | 模板渲染类型（基于物理渲染） |
| `reset_descriptors` | bool | False | 是否重新计算描述符 |
| `level_templates` | int | 0 | 模板密度级别（0=粗糙，1=中等，2=密集） |

### 分割器模型配置

#### SAM分割器 (segmentor_model/sam.yaml)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `_target_` | str | model.sam.CustomSamAutomaticMaskGenerator | SAM自动掩码生成器类路径 |
| `points_per_batch` | int | 64 | 每批处理的点数量 |
| `stability_score_thresh` | float | 0.85 | 稳定性分数阈值 |
| `box_nms_thresh` | float | 0.7 | 边界框NMS阈值 |
| `min_mask_region_area` | int | 0 | 最小掩码区域面积（像素） |
| `pred_iou_thresh` | float | 0.88 | 预测IoU阈值 |
| `sam.model_type` | str | vit_h | SAM模型类型（vit_h/vit_l/vit_b） |
| `sam.checkpoint_dir` | str | ./checkpoints/segment-anything/ | SAM模型检查点目录 |

#### YOLOv11分割器 (segmentor_model/yolov11seg.yaml)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `_target_` | str | model.yolov11_seg.Yolov11Seg | YOLOv11分割器类路径 |
| `checkpoint_path` | str | ./checkpoints/yolov11/best.pt | 模型权重文件路径 |
| `config.iou_threshold` | float | 0.5 | IoU阈值，用于NMS |
| `config.conf_threshold` | float | 0.25 | 置信度阈值 |
| `config.max_det` | int | 200 | 最大检测数量 |

#### FastSAM分割器 (segmentor_model/fast_sam.yaml)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `_target_` | str | model.fast_sam.CustomFastSAM | FastSAM分割器类路径 |
| `checkpoint_path` | str | ./checkpoints/FastSAM/ | FastSAM模型检查点目录 |
| `model_type` | str | FastSAM-x | FastSAM模型类型 |

### 描述符模型配置

#### DINOv2描述器 (descriptor_model/dinov2.yaml)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `_target_` | str | model.dinov2.CustomDINOv2 | DINOv2描述器类路径 |
| `model_name` | str | dinov2_vitl14 | DINOv2模型名称（ViT-L/14） |
| `checkpoint_dir` | str | ./checkpoints/dinov2 | DINOv2模型检查点目录 |
| `token_name` | str | x_norm_clstoken | 使用的token类型 |
| `image_size` | int | 224 | 输入图像尺寸 |
| `chunk_size` | int | 16 | 分块处理大小 |
| `validpatch_thresh` | float | 0.5 | 有效patch阈值 |

### 姿态估计模型配置 (PEM)

#### 优化器配置 (optimizer)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `type` | str | Adam | 优化器类型 |
| `lr` | float | 0.0001 | 学习率 |
| `betas` | list | [0.5, 0.999] | Adam优化器的beta参数 |
| `eps` | float | 0.000001 | 数值稳定性参数 |
| `weight_decay` | float | 0.0 | 权重衰减系数 |

#### 学习率调度器 (lr_scheduler)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `type` | str | WarmupCosineLR | 学习率调度器类型 |
| `max_iters` | int | 600000 | 最大迭代次数 |
| `warmup_factor` | float | 0.001 | 预热因子 |
| `warmup_iters` | int | 1000 | 预热迭代次数 |

#### 模型架构配置 (model)

**特征提取模块 (feature_extraction)**
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `vit_type` | str | vit_base | Vision Transformer类型 |
| `up_type` | str | linear | 上采样类型 |
| `embed_dim` | int | 768 | 嵌入维度 |
| `out_dim` | int | 256 | 输出特征维度 |
| `use_pyramid_feat` | bool | True | 是否使用金字塔特征 |
| `pretrained` | bool | True | 是否使用预训练权重 |

**几何嵌入模块 (geo_embedding)**
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `sigma_d` | float | 0.2 | 距离嵌入的标准差 |
| `sigma_a` | float | 15 | 角度嵌入的标准差 |
| `angle_k` | int | 3 | 角度嵌入的k近邻数 |
| `reduction_a` | str | max | 角度特征的聚合方式 |
| `hidden_dim` | int | 256 | 隐藏层维度 |

**粗粒度点匹配 (coarse_point_matching)**
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `nblock` | int | 3 | Transformer块数量 |
| `input_dim` | int | 256 | 输入特征维度 |
| `hidden_dim` | int | 256 | 隐藏层维度 |
| `out_dim` | int | 256 | 输出特征维度 |
| `temp` | float | 0.1 | 温度参数，用于softmax |
| `sim_type` | str | cosine | 相似度计算类型 |
| `normalize_feat` | bool | True | 是否归一化特征 |
| `loss_dis_thres` | float | 0.15 | 距离损失阈值 |
| `nproposal1` | int | 6000 | 第一阶段候选数量 |
| `nproposal2` | int | 300 | 第二阶段候选数量 |

**细粒度点匹配 (fine_point_matching)**
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `nblock` | int | 3 | Transformer块数量 |
| `input_dim` | int | 256 | 输入特征维度 |
| `hidden_dim` | int | 256 | 隐藏层维度 |
| `out_dim` | int | 256 | 输出特征维度 |
| `pe_radius1` | float | 0.1 | 第一阶段位置编码半径 |
| `pe_radius2` | float | 0.2 | 第二阶段位置编码半径 |
| `focusing_factor` | int | 3 | 聚焦因子 |
| `temp` | float | 0.1 | 温度参数 |
| `sim_type` | str | cosine | 相似度计算类型 |
| `normalize_feat` | bool | True | 是否归一化特征 |
| `loss_dis_thres` | float | 0.15 | 距离损失阈值 |

#### 数据集配置

**训练数据集 (train_dataset)**
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `data_dir` | str | ../Data/MegaPose-Training-Data | 训练数据目录 |
| `img_size` | int | 224 | 输入图像尺寸 |
| `n_sample_observed_point` | int | 2048 | 观测点采样数量 |
| `n_sample_model_point` | int | 2048 | 模型点采样数量 |
| `n_sample_template_point` | int | 5000 | 模板点采样数量 |
| `min_visib_fract` | float | 0.1 | 最小可见性比例 |
| `min_px_count_visib` | int | 512 | 最小可见像素数量 |
| `shift_range` | float | 0.01 | 数据增强的平移范围 |
| `rgb_mask_flag` | bool | True | 是否使用RGB掩码 |
| `dilate_mask` | bool | True | 是否膨胀掩码 |

**测试数据集 (test_dataset)**
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `data_dir` | str | ../Data/BOP | 测试数据目录 |
| `template_dir` | str | ../Data/BOP-Templates | 模板目录 |
| `img_size` | int | 224 | 输入图像尺寸 |
| `n_sample_observed_point` | int | 2048 | 观测点采样数量 |
| `n_sample_model_point` | int | 1024 | 模型点采样数量 |
| `n_sample_template_point` | int | 5000 | 模板点采样数量 |
| `minimum_n_point` | int | 8 | 最小点数量 |
| `rgb_mask_flag` | bool | True | 是否使用RGB掩码 |
| `seg_filter_score` | float | 0.25 | 分割过滤分数 |
| `n_template_view` | int | 42 | 模板视图数量 |

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

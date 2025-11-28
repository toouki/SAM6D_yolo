import os
import sys
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import trimesh
from PIL import Image
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
import time

# 添加ISM相关路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Instance_Segmentation_Model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pose_Estimation_Model'))

# ISM相关导入
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.model.utils import Detections
from Instance_Segmentation_Model.utils.inout import load_json, save_torch, load_torch

# 添加PEM模型路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pose_Estimation_Model', 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pose_Estimation_Model', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pose_Estimation_Model', 'model', 'pointnet2'))

# PEM相关导入
import gorilla
import importlib
import cv2
import imageio.v2 as imageio
from Pose_Estimation_Model.utils.draw_utils import draw_detections
import pycocotools.mask as cocomask
import torchvision.transforms as transforms
import colorsys
import distinctipy

# 常量定义
DEFAULT_MODEL_POINTS = 2048  # CAD模型采样点云的默认点数
DEFAULT_TEMPLATE_VIEWS = 42    # 姿态估计模板的总视图数（42个均匀分布的视角）
DEFAULT_NMS_ROTATION_THRESHOLD = 3.0    # 姿态非极大值抑制的旋转误差阈值（度）
DEFAULT_NMS_TRANSLATION_THRESHOLD = 5.0 # 姿态非极大值抑制的平移误差阈值（毫米）
MAX_INSTANCES = 3             # 单张图像中最多检测的物体实例数
MIN_MASK_PIXELS = 32           # 有效掩码的最小像素数，用于过滤过小的检测区域

# 设置日志（初始级别为INFO，可通过verbose参数动态调整）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RGB图像预处理
rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_distinct_colors(n_colors: int) -> List[tuple]:
    """
    生成视觉上可区分的颜色列表
    
    Args:
        n_colors: 需要的颜色数量
        
    Returns:
        颜色列表，每个颜色为(R,G,B)元组，值范围0-255
    """
    if n_colors <= 0:
        return []
    
    try:
        # 优先使用distinctipy库生成视觉上可区分的颜色
        colors = distinctipy.get_colors(n_colors)
        # 转换为0-255范围的RGB元组
        return [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
    except ImportError:
        # 如果distinctipy不可用，使用HSV色彩空间生成颜色
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.8 + (i % 2) * 0.2  # 交替使用0.8和1.0的饱和度
            value = 0.8 + (i % 3) * 0.067   # 在0.8-1.0之间变化亮度
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
        return colors


class PoseEstimationDetector:
    """
    SAM6D姿态估计检测器
    该类集成了实例分割模型(ISM)和姿态估计模型(PEM)，
    实现从RGB-D图像到6D物体姿态估计的完整流程。
    """
    
    def __init__(self, 
                segmentor_model: str = "sam", 
                pem_checkpoint: Optional[str] = None,
                cam_path: Optional[str] = None, 
                cad_path: Optional[str] = None, 
                output_dir: Optional[str] = None, 
                det_score_thresh: float = 0.2, 
                reset_descriptors: bool = True, 
                visualization: bool = False,
                verbose: bool = False,
                save_results: bool = True) -> None:
        """
        初始化姿态估计检测器
        
        Args:
            segmentor_model: 分割模型类型 ('sam', 'fastsam', 'sam2', 'yolo')
            pem_checkpoint: 姿态估计模型权重文件路径
            cam_path: 相机参数文件路径
            cad_path: CAD模型文件路径
            output_dir: 输出目录路径
            det_score_thresh: 检测分数阈值
            reset_descriptors: 是否重新计算描述符
            visualization: 是否启用可视化功能
            verbose: 是否启用详细日志输出
            save_results: 是否保存中间结果文件，False可加速计算
        """
        self.segmentor_model = segmentor_model
        self.pem_checkpoint = pem_checkpoint or os.path.join(
            os.path.dirname(__file__), 'Pose_Estimation_Model', 'checkpoints', 'sam-6d-pem-base.pth'
        )
        
        # 存储固定参数
        self.cam_path = cam_path
        self.cad_path = cad_path
        self.output_dir = output_dir
        self.det_score_thresh = det_score_thresh
        self.reset_descriptors = reset_descriptors
        self.visualization = visualization
        self.verbose = verbose
        self.save_results = save_results
        
        # 根据verbose参数设置日志级别
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ism_model = None
        self.pem_model = None
        self.pem_config = None
        self.ref_data = {}
        self.cam_info = None
        self.model_points = None
        
        # 姿态估计模板相关缓存
        self.all_tem = None
        self.all_tem_pts = None
        self.all_tem_feat = None
        
        # 图像数据缓存（避免重复读取）
        self._current_rgb = None
        self._current_depth = None
        self._current_whole_pts = None
        self._current_image_data = None
        
        # 如果提供了相机参数路径，预先加载
        if self.cam_path:
            self.cam_info = load_json(self.cam_path)
            
        # 如果提供了CAD模型路径，预先加载并采样点云
        if self.cad_path:
            mesh = trimesh.load_mesh(self.cad_path)
            self.model_points = mesh.sample(DEFAULT_MODEL_POINTS).astype(np.float32) / 1000.0  # 转为米单位
        
        if self.verbose:
            logger.info(f"初始化SAM6D检测器，分割模型: {segmentor_model}, 设备: {self.device}")
        
        self.initialize_models()
    
    def initialize_models(self) -> None:
        """
        初始化所有模型
        
        该方法可以用于：
        1. 在构造函数中立即初始化模型（当init_models_immediately=True时）
        2. 在需要时手动初始化模型，避免第一次推理时的延迟
        """
        if self.ism_model is None:
            self.init_ism_model(self.output_dir, self.reset_descriptors)
        if self.pem_model is None:
            self.init_pem_model()
        
        # 初始化姿态估计模板
        self._init_pose_templates()
        
        if self.verbose:
            logger.info("所有模型初始化完成")
    
    def _init_pose_templates(self) -> None:
        """
        初始化姿态估计模板数据
        
        预先加载和处理模板，避免每次推理时重复计算
        """
        if not self.cad_path or not self.pem_config:
            logger.warning("缺少CAD路径或PEM配置，跳过模板初始化")
            return
            
        # 查找模板目录
        tem_path = os.path.join(os.path.dirname(self.cad_path), 'templates')
        if not os.path.exists(tem_path):
            tem_path = os.path.join(os.path.dirname(self.cad_path), 'outputs', 'templates')
        
        if not os.path.exists(tem_path):
            logger.warning(f"模板目录不存在: {tem_path}")
            return
        
        if self.verbose:
            logger.info(f"加载姿态估计模板: {tem_path}")
        
        try:
            # 加载模板数据
            self.all_tem, self.all_tem_pts, self.all_tem_choose = self._init_templates(
                tem_path, self.pem_config.test_dataset
            )
            
            # 预计算模板特征
            with torch.no_grad():
                self.all_tem_pts, self.all_tem_feat = self.pem_model.feature_extraction.get_obj_feats(
                    self.all_tem, self.all_tem_pts, self.all_tem_choose
                )
            
            if self.verbose:
                logger.info(f"模板加载完成，共 {len(self.all_tem)} 个视图")
            
        except Exception as e:
            logger.error(f"模板加载失败: {e}")
            self.all_tem = None
            self.all_tem_pts = None
            self.all_tem_feat = None
    
    def init_ism_model(self, output_dir, reset_descriptors=False):
        """
        初始化实例分割模型(ISM)
        
        Args:
            output_dir: 输出目录，用于保存描述符
            cad_path: CAD模型路径，用于渲染模板
            reset_descriptors: 是否强制重新计算描述符
        """
        if self.verbose:
            logger.info("初始化ISM模型...")
        
        # 初始化Hydra配置
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs"):
            cfg = compose(config_name='run_inference.yaml')
        
        # 加载分割模型配置
        model_config_map = {
            "sam": "ISM_sam.yaml",
            "fastsam": "ISM_fastsam.yaml", 
            "sam2": "ISM_sam2.yaml",
            "yolo": "ISM_yolo.yaml"
        }
        
        if self.segmentor_model not in model_config_map:
            raise ValueError(f"不支持的分割模型: {self.segmentor_model}")
        
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
            cfg.model = compose(config_name=model_config_map[self.segmentor_model])
        
        # 实例化模型
        self.ism_model = instantiate(cfg.model)
        
        # 移动模型到设备
        self.ism_model.descriptor_model.model = self.ism_model.descriptor_model.model.to(self.device)
        self.ism_model.descriptor_model.model.device = self.device
        
        if hasattr(self.ism_model.segmentor_model, "predictor"):
            self.ism_model.segmentor_model.predictor.model = self.ism_model.segmentor_model.predictor.model.to(self.device)
        elif hasattr(self.ism_model.segmentor_model.model, "to"):
            self.ism_model.segmentor_model.model = self.ism_model.segmentor_model.model.to(self.device)
        else:
            self.ism_model.segmentor_model.model.setup_model(device=self.device, verbose=True)
        
        # 加载或计算模板描述符
        self._load_or_compute_descriptors(output_dir, reset_descriptors)
        
        # 将参考数据设置到ISM模型中
        self.ism_model.ref_data = self.ref_data
        
        self.ism_config = cfg
        if self.verbose:
            logger.info("ISM模型初始化完成")
    
    def init_pem_model(self, config_path=None):
        """
        初始化姿态估计模型(PEM)
        
        Args:
            config_path: PEM配置文件路径
        """
        if self.verbose:
            logger.info("初始化PEM模型...")
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'Pose_Estimation_Model', 'config', 'base.yaml')
        
        # 加载配置
        self.pem_config = gorilla.Config.fromfile(config_path)
        
        # 设置默认的模型名称（因为配置文件中没有model_name字段）
        model_name = getattr(self.pem_config, 'model_name', 'pose_estimation_model')
        
        # 动态导入模型
        MODEL = importlib.import_module(model_name)
        self.pem_model = MODEL.Net(self.pem_config.model)
        
        # 移动到GPU并设置为评估模式
        self.pem_model = self.pem_model.cuda()
        self.pem_model.eval()
        
        # 加载权重
        if os.path.exists(self.pem_checkpoint):
            gorilla.solver.load_checkpoint(model=self.pem_model, filename=self.pem_checkpoint)
            if self.verbose:
                logger.info(f"已加载PEM权重: {self.pem_checkpoint}")
        else:
            logger.warning(f"PEM权重文件不存在: {self.pem_checkpoint}")
    
    def _get_template(self, path, cfg, tem_index=1):
        """获取单个模板的信息"""
        # 构建模板的RGB、掩码、点云路径
        rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
        mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
        xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

        # 加载模板RGB图像并转为uint8格式
        rgb = self._load_im(rgb_path).astype(np.uint8)
        # 加载模板点云并转为float32，单位从mm转为m（除以1000）
        xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
        # 加载模板掩码并转为布尔值（255为前景）
        mask = self._load_im(mask_path).astype(np.uint8) == 255

        # 获取掩码的边界框
        bbox = self._get_bbox(mask)
        y1, y2, x1, x2 = bbox
        # 根据边界框裁剪掩码
        mask = mask[y1:y2, x1:x2]

        # 将RGB图像从BGR转为RGB，并根据边界框裁剪
        rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
        # 如果配置了RGB掩码，用掩码过滤RGB图像
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

        # 将裁剪后的RGB图像resize到配置的图像尺寸
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        # 应用预处理转换
        rgb = rgb_transform(np.array(rgb))

        # 获取掩码中非零区域的索引
        choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
        
        # 检查choose是否为空
        if len(choose) == 0:
            logger.warning("模板掩码中没有有效像素，使用随机采样")
            # 如果掩码为空，使用随机采样
            mask_flat = mask.flatten()
            valid_indices = np.arange(len(mask_flat))
            choose = np.random.choice(valid_indices, min(cfg.n_sample_template_point, len(valid_indices)), replace=False)
        
        # 采样模板点
        if len(choose) <= cfg.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
        choose = choose[choose_idx]
        # 根据边界框裁剪点云，展平后按采样索引选择点
        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

        # 获取Resize后RGB图像中对应采样点的索引
        rgb_choose = self._get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
        return rgb, rgb_choose, xyz

    def _init_templates(self, path, cfg):
        """获取所有模板"""
        n_template_view = cfg.n_template_view
        all_tem = []
        all_tem_choose = []
        all_tem_pts = []

        total_nView = DEFAULT_TEMPLATE_VIEWS
        for v in range(n_template_view):
            view_index = int(total_nView / n_template_view * v)
            tem, tem_choose, tem_pts = self._get_template(path, cfg, view_index)
            all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
            all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
            all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
        return all_tem, all_tem_pts, all_tem_choose


    
    def _get_det_data(self, rgb_path, depth_path, detections, det_score_thresh, cfg):
        """从内存中的检测结果获取测试数据（避免文件I/O）"""
        # 过滤掉分数低于阈值的检测结果
        dets = [det for det in detections if det['score'] > det_score_thresh]

        # 使用缓存数据（避免重复读取）
        image_data = self._load_image_data(rgb_path, depth_path)
        whole_image = image_data['rgb_array']
        whole_depth = image_data['depth_array'].astype(np.float32) * self.cam_info['depth_scale'] / 1000.0
        whole_pts = image_data['whole_pts']
        
        # 使用预加载的相机参数
        K = np.array(self.cam_info['cam_K']).reshape(3, 3)

        # 使用预加载的CAD模型点云
        if self.model_points is None:
            mesh = trimesh.load_mesh(self.cad_path)
            self.model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
        model_points = self.model_points
        radius = np.max(np.linalg.norm(model_points, axis=1))

        # 存储处理后的测试数据
        all_rgb = []
        all_cloud = []
        all_rgb_choose = []
        all_score = []
        all_dets = []

        # 遍历每个检测实例
        for inst in dets:
            seg = inst['segmentation']
            score = inst['score']

            # 处理掩码
            h, w = seg['size']  # 掩码尺寸（高度、宽度）
            try:
                # 将RLE格式转为掩码对象
                rle = cocomask.frPyObjects(seg, h, w)
            except:
                rle = seg  # 若转换失败，直接使用原始seg
            # 解码RLE得到掩码（二进制矩阵）
            mask = cocomask.decode(rle)
            mask = np.logical_and(mask > 0, whole_depth > 0)
            
            if np.sum(mask) > MIN_MASK_PIXELS:
                bbox = self._get_bbox(mask)
                y1, y2, x1, x2 = bbox
            else:
                continue

            # 根据边界框裁剪掩码
            mask = mask[y1:y2, x1:x2]
            # 获取掩码中非零区域的索引（展平后）
            choose = mask.astype(np.float32).flatten().nonzero()[0]
            
            # 检查choose是否为空
            if len(choose) == 0:
                logger.warning("掩码中没有有效像素，跳过该实例")
                continue

            # 处理点云
            cloud = whole_pts[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]
            
            # 计算点云中心
            center = np.mean(cloud, axis=0)
            # 点云去中心化
            tmp_cloud = cloud - center[None, :]
            # 过滤离群点
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            
            # 根据过滤结果更新索引和点云
            choose = choose[flag]
            cloud = cloud[flag]

            # 采样观测点（确保数量为配置的n_sample_observed_point）
            if len(choose) <= cfg.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]

            # 处理RGB图像
            rgb = whole_image[y1:y2, x1:x2, :][:,:,::-1]  # BGR to RGB
            if cfg.rgb_mask_flag:
                rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
            
            rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = rgb_transform(np.array(rgb))
            
            # 获取resize后RGB图像中对应采样点的索引
            rgb_choose = self._get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

            # 添加到列表
            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_score.append(score)
            all_dets.append(inst)

        # 构建输入数据
        input_data = {}
        if len(all_rgb) > 0:
            input_data['rgb'] = torch.stack(all_rgb).cuda()
            input_data['pts'] = torch.stack(all_cloud).cuda()
            input_data['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
            input_data['score'] = torch.FloatTensor(all_score).cuda()

            # 获取实例数量
            ninstance = len(all_rgb)
            # 复制模型点云，适配批次维度
            input_data['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
            # 复制相机内参，适配批次维度
            input_data['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()

        # 为每个检测结果添加相机内参
        for det in all_dets:
            det['K'] = K.tolist()  # 添加相机内参
        
        return input_data, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets
    
    def _load_or_compute_descriptors(self, output_dir, reset_descriptors):
        """加载或计算模板描述符"""
        descriptors_dir = os.path.join(output_dir, "descriptors")
        os.makedirs(descriptors_dir, exist_ok=True)
        
        main_desc_path = os.path.join(descriptors_dir, "main_descriptors.pth")
        appe_desc_path = os.path.join(descriptors_dir, "appe_descriptors.pth")
        
        # 加载模板数据
        template_dir = os.path.join(output_dir, 'templates')
        if not os.path.exists(template_dir):
            raise ValueError(f"模板目录不存在: {template_dir}")
        
        import glob
        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
            mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
            boxes.append(mask.getbbox())
            
            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
        
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        # 处理模板图像
        processing_config = OmegaConf.create({"image_size": 224})
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)
        
        # 初始化参考数据
        self.ref_data = {}
        
        # 主描述符
        if os.path.exists(main_desc_path) and not reset_descriptors:
            if self.verbose:
                logger.info(f"加载主描述符: {main_desc_path}")
            self.ref_data["descriptors"] = load_torch(main_desc_path).to(self.device)
        else:
            if self.verbose:
                logger.info("计算主描述符...")
            self.ref_data["descriptors"] = self.ism_model.descriptor_model.compute_features(
                templates, token_name="x_norm_clstoken"
            ).unsqueeze(0).data
            save_torch(self.ref_data["descriptors"], main_desc_path)
        
        # 外观描述符
        if os.path.exists(appe_desc_path) and not reset_descriptors:
            if self.verbose:
                logger.info(f"加载外观描述符: {appe_desc_path}")
            self.ref_data["appe_descriptors"] = load_torch(appe_desc_path).to(self.device)
        else:
            if self.verbose:
                logger.info("计算外观描述符...")
            self.ref_data["appe_descriptors"] = self.ism_model.descriptor_model.compute_masked_patch_feature(
                templates, masks_cropped[:, 0, :, :]
            ).unsqueeze(0).data
            save_torch(self.ref_data["appe_descriptors"], appe_desc_path)

    def _batch_input_data(self, depth_array, cam_path, device):
        """
        处理深度图和相机参数，转换为模型输入的批次数据格式
        
        参数:
            depth_array: 深度图像数组
            cam_path: 相机参数文件(json)路径
            device: 计算设备(cpu/cuda)
        返回:
            batch: 包含预处理后的数据字典，键包括'depth', 'cam_intrinsic', 'depth_scale'
        """
        batch = {}
        
        # 使用预加载的相机参数（避免重复读取）
        if self.cam_info is None:
            self.cam_info = load_json(cam_path)
        
        # 直接使用传入的深度数据
        depth_data = depth_array.astype(np.int32)
        
        # 解析相机内参(3x3矩阵)
        cam_K = np.array(self.cam_info['cam_K']).reshape((3, 3))
        # 获取深度缩放因子(将深度值转换为实际距离)
        depth_scale = np.array(self.cam_info['depth_scale'])

        # 转换为torch张量，增加批次维度，并移动到目标设备
        batch["depth"] = torch.from_numpy(depth_data).unsqueeze(0).to(device)
        batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
        batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
        return batch

    def _compute_rotation_error(self, R1, R2):
        """计算两个旋转矩阵的误差（度）"""
        # R1: [3,3], R2: [3,3]
        R_diff = R1.T @ R2  # 旋转差
        trace = np.trace(R_diff)
        trace = np.clip(trace, -1.0 + 1e-6, 3.0 - 1e-6)  # 数值稳定性处理
        angle_rad = np.arccos((trace - 1) / 2)
        return np.rad2deg(angle_rad)

    def _compute_translation_error(self, t1, t2):
        """计算两个平移向量的误差（mm）"""
        # t1: [3], t2: [3]
        return np.linalg.norm(t1 - t2)

    def _pose_nms(self, poses_rot, poses_trans, scores, rot_thresh, trans_thresh):
        """
        姿态非极大值抑制，去除重复姿态
        输入:
            poses_rot: [N, 3, 3] 旋转矩阵列表
            poses_trans: [N, 3] 平移向量列表（mm）
            scores: [N] 姿态分数
            rot_thresh: 旋转误差阈值（度）
            trans_thresh: 平移误差阈值（mm）
        输出:
            keep_indices: 保留的姿态索引
        """
        if len(poses_rot) == 0:
            return []
        
        # 按分数降序排序索引
        sorted_indices = np.argsort(scores)[::-1]
        keep = []
        
        for i in sorted_indices:
            current_rot = poses_rot[i]
            current_trans = poses_trans[i]
            is_duplicate = False
            
            # 与已保留的姿态比较
            for j in keep:
                rot_err = self._compute_rotation_error(current_rot, poses_rot[j])
                trans_err = self._compute_translation_error(current_trans, poses_trans[j])
                if rot_err < rot_thresh and trans_err < trans_thresh:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(i)
        
        return keep

    def _load_image_data(self, rgb_path, depth_path):
        """
        加载并缓存图像数据，避免重复读取
        
        Args:
            rgb_path: RGB图像路径
            depth_path: 深度图像路径
        """
        # 检查是否需要重新加载
        if (self._current_rgb is None or self._current_depth is None or 
            self._current_rgb_path != rgb_path or self._current_depth_path != depth_path):
            
            if self.verbose:
                logger.info(f"加载图像数据: RGB={rgb_path}, Depth={depth_path}")
            
            # 加载RGB图像
            self._current_rgb = Image.open(rgb_path).convert("RGB")
            
            # 加载深度图像
            self._current_depth = np.array(imageio.imread(depth_path)).astype(np.int32)
            
            # 计算点云（如果相机参数已加载）
            if self.cam_info:
                K = np.array(self.cam_info['cam_K']).reshape(3, 3)
                depth_scale = self.cam_info['depth_scale']
                depth_meters = self._current_depth.astype(np.float32) * depth_scale / 1000.0
                self._current_whole_pts = self._get_point_cloud_from_depth(depth_meters, K)
            
            # 缓存路径
            self._current_rgb_path = rgb_path
            self._current_depth_path = depth_path
            
            # 缓存预处理后的数据
            self._current_image_data = {
                'rgb_array': np.array(self._current_rgb),
                'depth_array': self._current_depth,
                'whole_pts': self._current_whole_pts
            }
        
        return self._current_image_data
    
    def _clear_image_cache(self):
        """清除图像缓存"""
        self._current_rgb = None
        self._current_depth = None
        self._current_whole_pts = None
        self._current_image_data = None
        self._current_rgb_path = None
        self._current_depth_path = None
    
    def _load_im(self, path):
        """加载图像"""
        return imageio.imread(path)

    def _get_bbox(self, label):
        """获取掩码的边界框"""
        # 处理多维掩码，如果是3D则取第一个通道
        if len(label.shape) == 3:
            label = label[:, :, 0]
        
        img_width, img_length = label.shape
        rows = np.any(label, axis=1)
        cols = np.any(label, axis=0)
        
        # 检查是否有有效的像素
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            # 没有有效像素，返回空边界框
            return 0, 0, 0, 0
        
        rmin, rmax = row_indices[[0, -1]]
        cmin, cmax = col_indices[[0, -1]]
        rmax += 1
        cmax += 1
        r_b = rmax - rmin
        c_b = cmax - cmin
        b = min(max(r_b, c_b), min(img_width, img_length))
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

        rmin = center[0] - int(b / 2)
        rmax = center[0] + int(b / 2)
        cmin = center[1] - int(b / 2)
        cmax = center[1] + int(b / 2)

        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_width:
            delt = rmax - img_width
            rmax = img_width
            rmin -= delt
        if cmax > img_length:
            delt = cmax - img_length
            cmax = img_length
            cmin -= delt

        return rmin, rmax, cmin, cmax

    def _get_point_cloud_from_depth(self, depth, K, bbox=None):
        """从深度图生成点云"""
        cam_fx, cam_fy, cam_cx, cam_cy = K[0,0], K[1,1], K[0,2], K[1,2]

        im_H, im_W = depth.shape
        xmap = np.array([[col_idx for col_idx in range(im_W)] for _ in range(im_H)])
        ymap = np.array([[row_idx for _ in range(im_W)] for row_idx in range(im_H)])

        if bbox is not None:
            rmin, rmax, cmin, cmax = bbox
            depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
            xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
            ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

        pt2 = depth.astype(np.float32)
        pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
        pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy

        cloud = np.stack([pt0,pt1,pt2]).transpose((1,2,0))
        return cloud

    def _get_resize_rgb_choose(self, choose, bbox, img_size):
        """获取resize后RGB图像的采样索引"""
        rmin, rmax, cmin, cmax = bbox
        crop_h = rmax - rmin
        ratio_h = img_size / crop_h
        crop_w = cmax - cmin
        ratio_w = img_size / crop_w

        row_idx = choose // crop_w
        col_idx = choose % crop_w
        choose = (np.floor(row_idx * ratio_h) * img_size + np.floor(col_idx * ratio_w)).astype(np.int64)
        return choose
        
    def run_instance_segmentation(self, rgb_array, depth_array):
        """
        运行实例分割
        
        Args:
            rgb_array: RGB图像数组
            depth_array: 深度图像数组
            
        Returns:
            detections: 检测结果
        """
        if self.verbose:
            logger.info("运行实例分割")
        
        # 检查必要参数
        if not self.cam_path:
            raise ValueError("初始化时必须提供cam_path参数")
        if not self.cad_path:
            raise ValueError("初始化时必须提供cad_path参数") 
        if not self.output_dir:
            raise ValueError("初始化时必须提供output_dir参数")
        
        # 生成掩码
        detections = self.ism_model.segmentor_model.generate_masks(rgb_array)
        detections = Detections(detections)
        
        # 过滤小目标
        detections.remove_very_small_detections(self.ism_config.model.post_processing_config.mask_post_processing)
        
        # 计算查询描述符
        query_descriptors, query_appe_descriptors = self.ism_model.descriptor_model.forward(rgb_array, detections)
        
        # 计算语义分数
        idx_selected_proposals, pred_idx_objects, semantic_score, best_template = self.ism_model.compute_semantic_score(query_descriptors)
        
        # 过滤检测结果
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
        
        # 计算外观分数
        appe_scores, ref_aux_descriptor = self.ism_model.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )
        
        # 处理深度和相机数据，准备几何分数计算
        batch = self._batch_input_data(depth_array, self.cam_path, self.device)
        # 获取模板的姿态分布(用于投影计算)
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4  # 缩放平移分量(可能是单位转换)
        # 转换姿态为tensor并移动到设备
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        # 加载level2中的姿态索引
        self.ism_model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

        # 使用预加载的CAD模型点云
        if self.model_points is None:
            raise ValueError("CAD模型未正确加载")
        self.ism_model.ref_data["pointcloud"] = torch.tensor(self.model_points).unsqueeze(0).data.to(self.device)
        
        # 将模板投影到图像平面，得到像素坐标
        image_uv = self.ism_model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        # 计算几何分数：通过点云投影与掩码的匹配度评估
        geometric_score, visible_ratio = self.ism_model.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.ism_model.visible_thred
        )

        # 综合分数计算：语义分数 + 外观分数 + 几何分数*可见比例，归一化
        final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)
        
        # 添加必要的属性
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))
        
        # 执行NMS
        detections.apply_nms(nms_thresh=self.ism_config.model.post_processing_config.nms_thresh)
        
        return detections, query_appe_descriptors, best_template, pred_idx_objects, ref_aux_descriptor
    

    
    def _get_det_data(self, rgb_array, depth_array, whole_pts, detections, det_score_thresh, cfg):
        """直接从传入的数组和ISM的Detections对象获取测试数据（避免文件I/O和格式转换）"""
        # 过滤掉分数低于阈值的检测结果
        valid_indices = []
        if hasattr(detections, 'scores'):
            scores = detections.scores.cpu().numpy()
            valid_indices = [i for i, score in enumerate(scores) if score > det_score_thresh]
        else:
            # 如果没有分数，保留所有检测
            valid_indices = list(range(len(detections)))

        # 直接使用传入的数据
        whole_image = rgb_array
        whole_depth = depth_array.astype(np.float32) * self.cam_info['depth_scale'] / 1000.0
        
        # 使用预加载的相机参数
        K = np.array(self.cam_info['cam_K']).reshape(3, 3)

        # 使用预加载的CAD模型点云
        if self.model_points is None:
            mesh = trimesh.load_mesh(self.cad_path)
            self.model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
        model_points = self.model_points
        radius = np.max(np.linalg.norm(model_points, axis=1))

        # 存储处理后的测试数据
        all_rgb = []
        all_cloud = []
        all_rgb_choose = []
        all_score = []
        all_dets = []

        # 遍历每个有效检测实例
        for i in valid_indices:
            # 直接从Detections对象获取数据
            if hasattr(detections, 'masks') and hasattr(detections, 'scores'):
                mask = detections.masks[i].cpu().numpy()
                score = float(detections.scores[i].cpu().numpy())
                bbox = detections.boxes[i].cpu().numpy() if hasattr(detections, 'boxes') else None
            else:
                continue

            # 直接使用二值掩码，无需RLE转换
            mask = np.logical_and(mask > 0, whole_depth > 0)
            
            if np.sum(mask) > MIN_MASK_PIXELS:
                bbox_coords = self._get_bbox(mask)
                y1, y2, x1, x2 = bbox_coords
            else:
                continue

            # 根据边界框裁剪掩码
            mask = mask[y1:y2, x1:x2]
            # 获取掩码中非零区域的索引（展平后）
            choose = mask.astype(np.float32).flatten().nonzero()[0]
            
            # 检查choose是否为空
            if len(choose) == 0:
                logger.warning("掩码中没有有效像素，跳过该实例")
                continue

            # 处理点云
            cloud = whole_pts[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]
            
            # 计算点云中心
            center = np.mean(cloud, axis=0)
            # 点云去中心化
            tmp_cloud = cloud - center[None, :]
            # 过滤离群点
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            
            # 根据过滤结果更新索引和点云
            choose = choose[flag]
            cloud = cloud[flag]

            # 采样观测点（确保数量为配置的n_sample_observed_point）
            if len(choose) <= cfg.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]

            # 处理RGB图像
            rgb = whole_image[y1:y2, x1:x2, :][:,:,::-1]  # BGR to RGB
            if cfg.rgb_mask_flag:
                rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
            
            rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = rgb_transform(np.array(rgb))
            
            # 获取resize后RGB图像中对应采样点的索引
            rgb_choose = self._get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

            # 创建检测结果字典（保持与原始格式兼容）
            detection_dict = {
                'score': score,
                'segmentation': {'size': mask.shape, 'counts': None},  # 标记格式
                'bbox': bbox.tolist() if bbox is not None else [x1, y1, x2-x1, y2-y1],
                'K': K.tolist()  # 添加相机内参
            }

            # 添加到列表
            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_score.append(score)
            all_dets.append(detection_dict)

        # 构建输入数据
        input_data = {}
        if len(all_rgb) > 0:
            input_data['rgb'] = torch.stack(all_rgb).cuda()
            input_data['pts'] = torch.stack(all_cloud).cuda()
            input_data['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
            input_data['score'] = torch.FloatTensor(all_score).cuda()

            # 获取实例数量
            ninstance = len(all_rgb)
            # 复制模型点云，适配批次维度
            input_data['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
            # 复制相机内参，适配批次维度
            input_data['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        
        return input_data, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets

    def run_pose_estimation(self, rgb_array, depth_array, whole_pts, detections, det_score_thresh=0.2):
        """
        从ISM检测结果运行姿态估计（合并后的完整方法）
        
        Args:
            rgb_array: RGB图像数组
            depth_array: 深度图像数组
            whole_pts: 完整点云数据
            detections: ISM的Detections对象
            det_score_thresh: 检测分数阈值
            
        Returns:
            results: 姿态估计结果
        """
        if self.verbose:
            logger.info("运行姿态估计...")
        
        # 检查必要参数
        if not self.cam_path or not self.cad_path:
            raise ValueError("初始化时必须提供cam_path, cad_path参数")
        
        # 直接使用传入的数据，避免重复读取
        input_data, img, _, model_points, detections_list = self._get_det_data(
            rgb_array, depth_array, whole_pts, detections, det_score_thresh, self.pem_config.test_dataset
        )
        
        # 姿态估计核心逻辑
        ninstance = input_data['pts'].size(0) if 'pts' in input_data else 0
        if ninstance == 0:
            logger.warning("没有找到有效的实例")
            return [], img, model_points * 1000  # 返回空正确的格式
        
        # 使用预加载的模板特征进行推理
        with torch.no_grad():
            input_data['dense_po'] = self.all_tem_pts.expand(ninstance, -1, -1)
            input_data['dense_fo'] = self.all_tem_feat.expand(ninstance, -1, -1)
            out = self.pem_model(input_data)
        
        # 计算姿态分数
        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] * out['score']
        else:
            pose_scores = out['score']
        
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000  # 转为毫米
        
        # 应用姿态NMS
        # 使用默认值
        rot_thresh = getattr(self.pem_config, 'rot_thresh', DEFAULT_NMS_ROTATION_THRESHOLD)
        trans_thresh = getattr(self.pem_config, 'trans_thresh', DEFAULT_NMS_TRANSLATION_THRESHOLD)
        keep_indices = self._pose_nms(
            poses_rot=pred_rot,
            poses_trans=pred_trans,
            scores=pose_scores,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh
        )
        
        # 限制最多检测实例数
        keep_indices = keep_indices[:MAX_INSTANCES]
        
        # 筛选结果
        results = []
        for i, idx in enumerate(keep_indices):
            result = {
                'instance_id': i,
                'score': float(pose_scores[idx]),
                'rotation': pred_rot[idx].tolist(),
                'translation': pred_trans[idx].tolist(),
                'bbox': detections_list[idx]['bbox'],
                'detection': detections_list[idx]
            }
            results.append(result)
        
        if self.verbose:
            logger.info(f"姿态估计完成，检测到 {len(results)} 个实例")
        return results, img, model_points  # 保持米为单位，在可视化时再转换
    

    
    def detect(self, rgb_path: str, depth_path: str) -> List[Dict[str, Any]]:
        """
        完整的6D物体姿态检测流程（仅使用内存方式）
        
        Args:
            rgb_path: RGB图像路径
            depth_path: 深度图像路径
            stability_score_thresh: SAM稳定性分数阈值
            
        Returns:
            results: 检测和姿态估计结果
        """
        if self.verbose:
            logger.info("开始完整的6D姿态检测流程（内存方式）...")
        
        # 检查必要参数
        if not self.output_dir:
            raise ValueError("初始化时必须提供output_dir参数")
        
        # 创建输出目录
        os.makedirs(f"{self.output_dir}", exist_ok=True)
        
        # 初始化模型（如果尚未初始化）
        if self.ism_model is None:
            self.init_ism_model(self.output_dir, self.reset_descriptors)
        if self.pem_model is None:
            self.init_pem_model()
        
        # 一次性加载所有图像数据，避免后续重复读取
        image_data = self._load_image_data(rgb_path, depth_path)
        rgb_array = image_data['rgb_array']
        depth_array = image_data['depth_array']
        whole_pts = image_data['whole_pts']
        
        # 步骤1: 实例分割（传入RGB数组而非路径）
        detections, _, _, _, _ = self.run_instance_segmentation(rgb_array, depth_array)
        
        if len(detections) == 0:
            logger.warning("ISM未检测到任何物体")
            return []
        
        # 步骤2: 姿态估计（传入预处理的数据而非路径）
        pose_results, img, model_points = self.run_pose_estimation(
            rgb_array, depth_array, whole_pts, detections, self.det_score_thresh
        )
        
        # 步骤3: 可视化结果（如果启用）
        if self.visualization:
            # 可视化实例分割结果
            self._visualize_seg(rgb_array, detections, self.output_dir)
            # 可视化姿态估计结果
            self._visualize_poses(img, pose_results, model_points, self.output_dir)
        else:
            if self.verbose:
                logger.info("可视化功能已禁用，跳过可视化步骤")
        
        if self.verbose:
            logger.info(f"6D姿态检测完成，共检测到 {len(pose_results)} 个物体实例")
        return pose_results
    
    def _visualize_poses(self, rgb_image, pose_results, model_points, output_dir=None):
        """统一的姿态估计结果可视化函数
        
        Args:
            rgb_image: RGB图像数组或PIL图像
            pose_results: 姿态估计结果列表
            model_points: CAD模型点云数据
            output_dir: 输出目录路径，如果为None则不保存结果
            
        Returns:
            concat_img: 拼接后的可视化图像，如果失败返回None
        """
        if not pose_results:
            if self.verbose:
                logger.warning("没有姿态估计结果，跳过可视化")
            return None
        
        # 确保输入格式正确
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)
        
        # 获取相机内参
        if self.cam_info is None:
            logger.error("相机参数未加载，无法可视化")
            return None
        
        K = np.array(self.cam_info['cam_K']).reshape(3, 3)
        
        try:
            # 提取所有旋转和平移矩阵
            pred_rotations = []
            pred_translations = []
            
            for result in pose_results:
                pred_rot = np.array(result['rotation'])
                pred_trans = np.array(result['translation']) / 1000.0  # 转为米单位
                pred_rotations.append(pred_rot)
                pred_translations.append(pred_trans)
            
            # 确保RGB图像是uint8类型且范围在[0, 255]
            if rgb_image.dtype != np.uint8:
                rgb_image_display = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image_display = rgb_image.copy()
            
            # 为每个姿态估计结果生成足够的颜色
            num_poses = len(pose_results)
            colors = generate_distinct_colors(max(num_poses, 6))  # 至少生成6种颜色
            
            # 初始化绘制图像
            img_with_poses = rgb_image_display.copy()
            
            # 直接在图像上绘制每个姿态估计结果，不使用半透明
            for i, result in enumerate(pose_results):
                color = colors[i]
                
                # 直接在主图像上绘制当前姿态估计结果
                img_with_poses = draw_detections(
                    img_with_poses,
                    [np.array(result['rotation'])],
                    [np.array(result['translation']) / 1000.0],  # 转为米单位
                    model_points,
                    [K],
                    color=color
                )
            
            # 在姿态结果图像上添加分数标签
            for i, result in enumerate(pose_results):
                score = result['score']
                color = colors[i % len(colors)]
                label = f"Pose {i}: {score:.3f}"
                
                # 添加标签背景和文字
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # 右上角显示标签
                label_x = img_with_poses.shape[1] - w - 10
                label_y = 30 + i * 30
                # 使用半透明背景而不是黑色背景
                overlay = img_with_poses.copy()
                cv2.rectangle(overlay, (label_x - 5, label_y - h), (label_x + w + 5, label_y + 5), (255, 255, 255), -1)
                img_with_poses = cv2.addWeighted(img_with_poses, 0.7, overlay, 0.3, 0)
                cv2.putText(img_with_poses, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 不创建拼接图像，只使用姿态估计结果
            concat_img = img_with_poses
            
            # 保存结果（如果提供了输出目录）
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "pose_estimation.png")
                cv2.imwrite(save_path, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))
                
                if self.verbose:
                    logger.info(f"姿态估计可视化结果已保存: {save_path}")
            
            return concat_img
            
        except Exception as e:
            logger.error(f"姿态可视化失败: {e}")
            return None
    
    def _visualize_seg(self, rgb_array, detections, output_dir):
        """可视化实例分割结果"""
        if len(detections) == 0:
            if self.verbose:
                logger.warning("没有实例分割结果，跳过分割可视化")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建可视化图像
        vis_img = rgb_array.copy()
        
        # 确保RGB图像是uint8类型且范围在[0, 255]
        if vis_img.dtype != np.uint8:
            vis_img = (vis_img * 255).astype(np.uint8)
        
        # 获取掩码数量并生成足够的颜色
        num_detections = len(detections.masks)
        colors = generate_distinct_colors(max(num_detections, 6))  # 至少生成6种颜色
        
        for i in range(num_detections):
            color = colors[i]
            
            # 获取掩码
            mask = detections.masks[i].cpu().numpy()
            
            # 确保掩码是2D的
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # 使用更好的混合方式 - 先创建轮廓再叠加
            # 找到掩码的轮廓
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制轮廓
            cv2.drawContours(vis_img, contours, -1, color, 2)
            
            # 半透明填充掩码区域
            alpha = 0.3  # 降低透明度使图像更明亮
            overlay = vis_img.copy()
            overlay[mask > 0] = color
            vis_img = cv2.addWeighted(vis_img, 1 - alpha, overlay, alpha, 0)
            
            # 绘制边界框（如果有）
            if hasattr(detections, 'boxes'):
                bbox = detections.boxes[i].cpu().numpy()
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签
            score = float(detections.scores[i].cpu().numpy()) if hasattr(detections, 'scores') else 0.0
            label = f"Instance {i}: {score:.3f}"
            
            # 计算文本尺寸
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_x = 10
            label_y = 30 + i * 30
            
            # 使用半透明白色背景而不是黑色背景
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (label_x - 5, label_y - h), (label_x + w + 5, label_y + 5), (255, 255, 255), -1)
            vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
            cv2.putText(vis_img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 保存结果
        save_path = os.path.join(output_dir, "segmentation_result.png")
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        if self.verbose:
            logger.info(f"实例分割可视化结果已保存: {save_path}")
    



# 使用示例
if __name__ == "__main__":
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 记录完整初始化开始时间（包括模型初始化）
    init_start_time = time.time()
    
    # 创建检测器实例，在初始化时传入所有固定参数
    '''
    使用sam2时
    如果要切换s、b、l型号，需要修改SAM-6D/Instance_Segmentation_Model/configs/model/segmentor_model/sam2.yaml
    将checkpoint_path: ./Instance_Segmentation_Model/checkpoints/SAM2/sam2.1_b.pt  改为
        checkpoint_path: ./Instance_Segmentation_Model/checkpoints/SAM2/sam2.1_l.pt 就会自动下载l型号的模型
    其他模型也同理
    使用自己训练的yoloseg时
    best.pt放在./Instance_Segmentation_Model/checkpoints/yolo/best.pt
    有多种类别默认只会取第一个类别进行检测
    要修改第几类修改：SAM-6D/Instance_Segmentation_Model/model/yolo.py
    '''

    detector = PoseEstimationDetector(
        segmentor_model="sam2",
        cam_path=os.path.join(base_dir, "Data/Example/camera.json"),
        cad_path=os.path.join(base_dir, "Data/Example/obj_000005.ply"),
        output_dir=os.path.join(base_dir, "Data/Example/outputs"),
        det_score_thresh=0.2,
        reset_descriptors=True,
        visualization=False,
        verbose=True,
        save_results=False
    )
    
    # 计算初始化耗时
    init_time = time.time() - init_start_time
    print(f"初始化耗时: {init_time:.2f}秒")
    
    # 运行多次检测以评估性能
    for i in range(1):
        # 记录检测开始时间
        detect_start_time = time.time()
        
        # 运行检测，只需要传入图像和深度图
        results = detector.detect(
            rgb_path=os.path.join(base_dir, "Data/Example/rgb.png"),
            depth_path=os.path.join(base_dir, "Data/Example/depth.png")
        )
        # 计算检测耗时
        detect_time = time.time() - detect_start_time
        print(f"检测耗时: {detect_time:.2f}秒")
    
    for i, result in enumerate(results):
        print(f"实例 {i}: 分数={result['score']:.4f}, "
            f"旋转矩阵={result['rotation']}, 平移={result['translation']}")


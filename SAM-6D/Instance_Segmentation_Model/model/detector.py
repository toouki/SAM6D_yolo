"""
实例分割模型类，用于物体检测和姿态估计任务。
该类继承自PyTorch Lightning的LightningModule，整合了分割、特征提取、匹配和后处理模块。
"""

import torch
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import os
import os.path as osp
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from utils.inout import save_json, load_json, save_json_bop23
from model.utils import BatchedData, Detections, convert_npz_to_json
from hydra.utils import instantiate
import time
import glob
from functools import partial
import multiprocessing
import trimesh
from model.loss import MaskedPatch_MatrixSimilarity
from utils.trimesh_utils import depth_image_to_pointcloud_translate_torch
from utils.poses.pose_utils import get_obj_poses_from_template_level
from utils.bbox_utils import xyxy_to_xywh, compute_iou


class Instance_Segmentation_Model(pl.LightningModule):
    """
    实例分割模型，整合了分割器、描述符模型和匹配逻辑。
    用于从输入图像中检测对象、估计姿态并输出结果。
    """

    def __init__(
        self,
        segmentor_model,  # 分割模型 (例如 SAM)
        descriptor_model, # 特征描述符模型
        onboarding_config, # 初始化配置
        matching_config,   # 匹配配置
        post_processing_config, # 后处理配置
        log_interval,      # 日志记录间隔
        log_dir,           # 日志保存目录
        visible_thred,     # 可见性阈值
        pointcloud_sample_num, # 点云采样数量
        **kwargs,          # 其他参数
    ):
        """
        初始化模型，设置各种组件和配置。
        """
        super().__init__()
        # 保存模型组件
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        # 保存配置
        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir

        # 保存其他参数
        self.visible_thred = visible_thred
        self.pointcloud_sample_num = pointcloud_sample_num

        # 定义反向图像归一化变换（用于将归一化图像还原为原始像素值）
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], # 反向均值
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],                # 反向标准差
                ),
            ]
        )
        logging.info(f"Init CNOS done!")

    def set_reference_objects(self):
        """
        加载或计算参考对象的特征描述符和外观描述符。
        这些描述符用于后续的匹配过程。
        """
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        # 初始化存储参考数据的字典
        self.ref_data = {
            "descriptors": BatchedData(None),      # 用于语义匹配的特征描述符
            "appe_descriptors": BatchedData(None), # 用于外观匹配的特征描述符
        }
        # 定义描述符文件路径
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        appe_descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors_appe.pth")

        # --- 加载或计算主要特征描述符 ---
        # 如果使用PBR渲染，则调整路径
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        # 如果存在预计算的描述符文件且不强制重置，则加载
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
        else:
            # 遍历模板数据集，计算每个模板的特征描述符
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device) # N_templates x 3 x H x W
                # 计算特征，使用 CLS token 作为图像级描述符
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                ) # N_templates x descriptor_size
                self.ref_data["descriptors"].append(ref_feats)

            self.ref_data["descriptors"].stack()  # 将列表堆叠为张量: N_objects x N_templates x descriptor_size
            # 取第一个维度的数据 (假设每个物体只有一个模板的描述符，或者取平均/聚合)
            # 注意：这里可能需要根据实际情况调整，通常是 N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data 

            # 保存计算好的描述符，供后续使用
            torch.save(self.ref_data["descriptors"], descriptors_path)

        # --- 加载或计算外观特征描述符 ---
        if self.onboarding_config.rendering_type == "pbr":
            appe_descriptors_path = appe_descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(appe_descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path).to(self.device)
        else:
            # 遍历模板数据集，计算每个模板的外观特征描述符（使用掩码）
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing appearance descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device) # N_templates x 3 x H x W
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device) # N_templates x 1 x H x W
                # 计算掩码区域的特征
                ref_feats = self.descriptor_model.compute_masked_patch_feature(
                    ref_imgs, ref_masks
                ) # N_templates x N_patches x feature_size
                self.ref_data["appe_descriptors"].append(ref_feats)

            self.ref_data["appe_descriptors"].stack() # N_objects x N_templates x N_patches x feature_size
            self.ref_data["appe_descriptors"] = self.ref_data["appe_descriptors"].data # 取第一个维度数据

            # 保存计算好的外观描述符
            torch.save(self.ref_data["appe_descriptors"], appe_descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}, "
            f"Appearance descriptors shape: {self.ref_data['appe_descriptors'].shape}"
        )

    def set_reference_object_pointcloud(self):
        """
        加载或生成参考对象的点云数据。
        用于后续的几何匹配和姿态估计。
        """
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects point cloud ...")

        start_time = time.time()
        pointcloud = BatchedData(None) # 临时存储点云数据
        pointcloud_path = osp.join(self.ref_dataset.template_dir, "pointcloud.pth")
        obj_pose_path = f"{self.ref_dataset.template_dir}/template_poses.npy"

        # --- 加载或生成模板姿态 ---
        if (
            os.path.exists(obj_pose_path)
            and not self.onboarding_config.reset_descriptors
        ):
            # 加载已存在的姿态
            poses = torch.tensor(np.load(obj_pose_path)).to(self.device).to(torch.float32) # N_all_template x 4 x 4
        else:
            # 生成模板姿态（从预定义分布中采样）
            template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all") # 获取预定义姿态
            template_poses[:, :3, 3] *= 0.4 # 缩放平移部分
            poses = torch.tensor(template_poses).to(self.device).to(torch.float32)
            np.save(obj_pose_path, template_poses) # 保存生成的姿态

        # 根据当前数据集的索引选择对应的姿态
        self.ref_data["poses"] = poses[self.ref_dataset.index_templates, :, :] # N_template x 4 x 4

        # --- 加载或生成点云 ---
        if (
            os.path.exists(pointcloud_path)
            and not self.onboarding_config.reset_descriptors
        ):
            # 加载已存在的点云 (注意：map_location="cuda:0" 强制加载到cuda:0，然后移动到当前设备)
            self.ref_data["pointcloud"] = torch.load(pointcloud_path, map_location="cuda:0").to(self.device)
        else:
            # 检查CAD模型文件夹是否存在
            mesh_path = osp.join(self.ref_dataset.root_dir, "models")
            if not os.path.exists(mesh_path):
                raise Exception("Can not find the mesh path.")
            # 遍历参考数据集，从CAD模型生成点云
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Generating pointcloud dataset ...",
            ):
                # 加载CAD模型
                if self.dataset_name == "lmo": # LineMOD Occluded 特殊处理
                    all_pc_idx = [1, 5, 6, 8, 9, 10, 11, 12] # 对应的模型ID
                    pc_id = all_pc_idx[idx]
                else:
                    pc_id = idx + 1 # 一般情况：索引+1
                mesh = trimesh.load_mesh(os.path.join(mesh_path, f'obj_{(pc_id):06d}.ply')) # 加载PLY文件
                # 从网格中采样指定数量的点，并转换单位 (mm -> m)
                model_points = mesh.sample(self.pointcloud_sample_num).astype(np.float32) / 1000.0
                pointcloud.append(torch.tensor(model_points)) # N_pointcloud x 3

            pointcloud.stack()  # 堆叠: N_objects x N_pointcloud x 3
            self.ref_data["pointcloud"] = pointcloud.data.to(self.device)

            # 保存生成的点云
            torch.save(self.ref_data["pointcloud"], pointcloud_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Pointcloud shape: {self.ref_data['pointcloud'].shape}"
        )

    def best_template_pose(self, scores, pred_idx_objects):
        """
        从多个模板中为每个预测对象选择最佳匹配的模板。
        Args:
            scores: 评分张量 (N_proposals x N_objects x N_templates)
            pred_idx_objects: 预测的对象ID (N_selected_proposals)
        Returns:
            best_template_idx: 最佳模板索引 (N_selected_proposals)
        """
        _, best_template_idxes = torch.max(scores, dim=-1) # 在模板维度找最大值: (N_proposals x N_objects, N_proposals x N_objects)
        N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
        # 重复对象ID，使其与N_object维度对齐
        pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

        assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

        # 根据预测的对象ID索引，获取对应的最佳模板ID
        best_template_idx = torch.gather(best_template_idxes, dim=1, index=pred_idx_objects)[:, 0]

        return best_template_idx

    def project_template_to_image(self, best_pose, pred_object_idx, batch, proposals):
        """
        将最佳模板的姿态应用于参考点云，并将其投影到查询图像上。
        用于计算几何得分。
        Args:
            best_pose: 最佳模板姿态索引 (N_selected_proposals)
            pred_object_idx: 预测的对象ID (N_selected_proposals)
            batch: 当前批次数据 (包含深度图、内参等)
            proposals: 预测的掩码 (N_selected_proposals x H x W)
        Returns:
            image_vu: 投影后的像素坐标 (N_selected_proposals x N_pointcloud x 2)
        """
        # 获取最佳姿态的旋转矩阵
        pose_R = self.ref_data["poses"][best_pose, 0:3, 0:3] # N_query x 3 x 3
        # 选择对应对象的点云
        select_pc = self.ref_data["pointcloud"][pred_object_idx, ...] # N_query x N_pointcloud x 3
        (N_query, N_pointcloud, _) = select_pc.shape

        # 将点云旋转（应用姿态的旋转部分）
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1) # N_query x N_pointcloud x 3

        # 计算平移向量（从深度图和掩码计算）
        # proposals: (N_query, H, W), batch["depth"]: (H, W), batch["cam_intrinsic"]: (3, 3), batch['depth_scale']: scalar
        translate = self.Calculate_the_query_translation(proposals, batch["depth"][0], batch["cam_intrinsic"][0], batch['depth_scale'])
        # 应用平移
        posed_pc = posed_pc + translate[:, None, :].repeat(1, N_pointcloud, 1) # broadcast translate to each point

        # 投影到图像平面
        cam_instrinsic = batch["cam_intrinsic"][0][None, ...].repeat(N_query, 1, 1).to(torch.float32) # (N_query, 3, 3)
        # 齐次坐标投影
        image_homo = torch.bmm(cam_instrinsic, posed_pc.permute(0, 2, 1)).permute(0, 2, 1) # (N_query, N_pointcloud, 3)
        # 转换为像素坐标 (除以齐次坐标第三维)
        image_vu = (image_homo / image_homo[:, :, -1][:, :, None])[:, :, 0:2].to(torch.int) # (N_query, N_pointcloud, 2)

        (imageH, imageW) = batch["depth"][0].shape
        # 将坐标限制在图像边界内
        image_vu[:, :, 0].clamp_(min=0, max=imageW - 1) # U坐标
        image_vu[:, :, 1].clamp_(min=0, max=imageH - 1) # V坐标

        return image_vu

    def Calculate_the_query_translation(self, proposal, depth, cam_intrinsic, depth_scale):
        """
        根据分割掩码和深度图计算物体中心的平移向量。
        Args:
            proposal: 预测的掩码 (N_query x H x W)
            depth: 深度图 (H x W)
            cam_intrinsic: 相机内参矩阵 (3 x 3)
            depth_scale: 深度缩放因子
        Returns:
            translate: 平移向量 (N_query x 3)
        """
        (N_query, imageH, imageW) = proposal.squeeze_().shape
        # 使用掩码提取深度值
        masked_depth = proposal * (depth[None, ...].repeat(N_query, 1, 1)) # (N_query, H, W)
        # 调用工具函数计算平移（通常基于掩码内深度的平均值）
        translate = depth_image_to_pointcloud_translate_torch(
            masked_depth, depth_scale, cam_intrinsic
        )
        return translate.to(torch.float32)

    def move_to_device(self):
        """
        将模型组件移动到指定设备（GPU/CPU）。
        """
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device
        # 根据分割模型的类型，移动其内部模型
        if hasattr(self.segmentor_model, "predictor"):
            # 对于像Detectron2这样的模型
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        else:
            # 对于像SAM这样的模型
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def compute_semantic_score(self, proposal_decriptors):
        """
        计算查询提案与参考对象之间的语义相似度得分。
        Args:
            proposal_decriptors: 查询提案的特征描述符 (N_proposals x descriptor_size)
        Returns:
            idx_selected_proposals: 通过置信度阈值筛选的提案索引
            pred_idx_objects: 预测的对象ID
            semantic_score: 语义得分
            best_template: 最佳模板索引
        """
        # 计算匹配得分: N_proposals x N_objects x N_templates
        scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"]
        )
        # 聚合模板维度的得分 (例如，取平均值)
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1] # sum / num_templates
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0] # 取前5个
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # 为每个提案分配得分最高的对象ID
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_proposals (for each proposal)

        # 筛选得分高于阈值的提案
        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        semantic_score = score_per_proposal[idx_selected_proposals]

        # 基于筛选后的提案和分配的对象ID，计算最佳模板
        flitered_scores = scores[idx_selected_proposals, ...] # N_selected x N_objects x N_templates
        best_template = self.best_template_pose(flitered_scores, pred_idx_objects)

        return idx_selected_proposals, pred_idx_objects, semantic_score, best_template

    def compute_appearance_score(self, best_pose, pred_objects_idx, qurey_appe_descriptors):
        """
        基于最佳模板计算外观相似度得分。
        Args:
            best_pose: 最佳模板索引 (N_selected_proposals)
            pred_objects_idx: 预测的对象ID (N_selected_proposals)
            qurey_appe_descriptors: 查询提案的外观描述符 (N_selected_proposals x N_patches x feature_size)
        Returns:
            appe_scores: 外观得分 (N_selected_proposals)
            ref_appe_descriptors: 对应的参考外观描述符 (N_selected_proposals x N_patches x feature_size)
        """
        # 构造索引张量以从参考外观描述符中选择正确的项
        con_idx = torch.concatenate((pred_objects_idx[None, :], best_pose[None, :]), dim=0) # (2, N_selected)
        # 选择对应的参考外观描述符: N_selected x N_patches x feature_size
        ref_appe_descriptors = self.ref_data["appe_descriptors"][con_idx[0, ...], con_idx[1, ...], ...]

        # 使用相似度度量计算得分
        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(qurey_appe_descriptors, ref_appe_descriptors) # N_selected

        return appe_scores, ref_appe_descriptors

    def compute_geometric_score(self, image_uv, proposals, appe_descriptors, ref_aux_descriptor, visible_thred=0.5):
        """
        计算几何得分（IoU 和 可见性比率）。
        Args:
            image_uv: 投影后的点云像素坐标 (N_selected_proposals x N_pointcloud x 2)
            proposals: 预测的掩码 (N_selected_proposals x H x W)，用于获取边界框
            appe_descriptors: 查询外观描述符
            ref_aux_descriptor: 参考外观描述符
            visible_thred: 可见性阈值
        Returns:
            iou: 投影点云边界框与预测掩码边界框的IoU (N_selected_proposals)
            visible_ratio: 可见性比率 (N_selected_proposals)
        """
        # 使用外观描述符计算可见性比率
        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(appe_descriptors, ref_aux_descriptor, visible_thred)

        # 从投影点云计算边界框 (xyxy格式)
        y1x1 = torch.min(image_uv, dim=1).values # 最小的 (v, u) 坐标: (N_selected, 2)
        y2x2 = torch.max(image_uv, dim=1).values # 最大的 (v, u) 坐标: (N_selected, 2)
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1) # (N_selected, 4)

        # 计算投影边界框与预测掩码边界框的IoU
        iou = compute_iou(xyxy, proposals.boxes) # proposals.boxes 应该是 (N_selected, 4)

        return iou, visible_ratio

    def test_step(self, batch, idx):
        """
        测试步骤，处理单个批次的数据。
        """
        if idx == 0: # 第一个批次时，初始化参考数据和模型
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.set_reference_object_pointcloud()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1" # 确保批大小为1

        # 反向归一化图像以便可视化或输入到分割模型
        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # (3, H, W)
            .cpu()
            .numpy()
            .transpose(1, 2, 0) # (H, W, 3)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # --- 提案生成阶段 ---
        proposal_stage_start_time = time.time()
        # 使用分割模型生成候选区域（掩码）
        proposals = self.segmentor_model.generate_masks(image_np) # 返回掩码列表

        # 初始化检测结果
        detections = Detections(proposals)
        # 移除非常小的检测框
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )

        # 计算查询提案的特征描述符和外观描述符
        query_decriptors, query_appe_descriptors = self.descriptor_model(image_np, detections)
        proposal_stage_end_time = time.time()

        # --- 匹配阶段 ---
        matching_stage_start_time = time.time()
        # 计算语义得分并筛选提案
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.compute_semantic_score(query_decriptors)

        # 根据筛选结果更新检测结果和查询描述符
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # 计算外观得分
        appe_scores, ref_aux_descriptor = self.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        # 计算几何得分
        # 注意：这里传入的是 detections.masks (N_selected x H x W)，project_template_to_image需要这个格式
        image_uv = self.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        geometric_score, visible_ratio = self.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.visible_thred
        )

        # --- 计算最终得分 ---
        # 组合语义、外观和几何得分
        final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)

        # 将最终得分和对象ID添加到检测结果中
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        # 对每个对象ID执行NMS（非极大值抑制）
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        # 计算总运行时间
        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        # 转换为numpy格式以便保存
        detections.to_numpy()

        # 构建保存路径
        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # 保存检测结果到文件
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # 保存运行时间到文件
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_epoch_end(self, outputs):
        """
        测试 epoch 结束时的处理逻辑，将所有NPZ结果文件转换为JSON格式。
        """
        if self.global_rank == 0:  # 只在主进程中执行
            # 收集所有结果文件路径
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            # 排除运行时间文件
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
            # 使用多进程池并行转换
            pool = multiprocessing.Pool(processes=num_workers)
            convert_npz_to_json_with_idx = partial(
                convert_npz_to_json,
                list_npz_paths=result_paths,
            )
            detections = list(
                tqdm(
                    pool.imap_unordered(
                        convert_npz_to_json_with_idx, range(len(result_paths))
                    ),
                    total=len(result_paths),
                    desc="Converting npz to json",
                )
            )
            # 将多进程返回的列表展平
            formatted_detections = []
            for detection in tqdm(detections, desc="Loading results ..."):
                formatted_detections.extend(detection)

            # 保存最终的JSON结果文件
            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")
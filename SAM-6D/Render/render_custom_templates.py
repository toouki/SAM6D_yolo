import blenderproc as bproc
import os
import argparse
import cv2
import numpy as np
import trimesh
from typing import Optional, List, Tuple


class CADTemplateRenderer:
    """CAD模型模板渲染器，用于生成多角度的RGB图像、掩码和NOCS坐标"""
    
    def __init__(self, 
                 cam_poses_path: Optional[str] = None,
                 light_scale: float = 2.5,
                 light_energy: int = 1000,
                 max_samples: int = 50):
        """
        初始化渲染器
        
        Args:
            cam_poses_path: 预定义相机姿态文件路径
            light_scale: 光源位置缩放因子
            light_energy: 光照强度
            max_samples: 渲染最大采样数
        """
        self.light_scale = light_scale
        self.light_energy = light_energy
        self.max_samples = max_samples
        
        # 获取默认相机姿态路径
        if cam_poses_path is None:
            render_dir = os.path.dirname(os.path.abspath(__file__))
            cam_poses_path = os.path.join(
                render_dir, 
                '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy'
            )
        
        self.cam_poses = np.load(cam_poses_path)
        
        # 初始化BlenderProc环境
        bproc.init()
    
    def get_norm_info(self, mesh_path: str) -> float:
        """
        计算模型的归一化尺度因子
        
        Args:
            mesh_path: 3D模型文件路径
            
        Returns:
            float: 缩放因子
        """
        mesh = trimesh.load(mesh_path, force='mesh')
        model_points = trimesh.sample.sample_surface(mesh, 1024)[0].astype(np.float32)
        
        min_value = np.min(model_points, axis=0)
        max_value = np.max(model_points, axis=0)
        radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))
        
        return 1 / (2 * radius)
    
    def _setup_lighting(self, cam_pose: np.ndarray) -> None:
        """设置场景光照"""
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location([
            self.light_scale * cam_pose[:3, -1][0],
            self.light_scale * cam_pose[:3, -1][1], 
            self.light_scale * cam_pose[:3, -1][2]
        ])
        light.set_energy(self.light_energy)
    
    def _setup_camera(self, cam_pose: np.ndarray) -> None:
        """设置相机姿态"""
        # 调整坐标系以适应Blender
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
        cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
        bproc.camera.add_camera_pose(cam_pose)
    
    def _setup_material(self, obj, base_color: float) -> None:
        """为对象设置材质"""
        color = [base_color, base_color, base_color, 0.]
        material = bproc.material.create('obj')
        material.set_principled_shader_value('Base Color', color)
        obj.set_material(0, material)
    
    def _save_outputs(self, data: dict, idx: int, output_dir: str) -> None:
        """保存渲染输出"""
        save_fpath = os.path.join(output_dir, "templates")
        os.makedirs(save_fpath, exist_ok=True)
        
        # 保存RGB图像
        color_bgr = data["colors"][0]
        color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath, f'rgb_{idx}.png'), color_bgr)
        
        # 保存掩码
        mask = data["nocs"][0][..., -1]
        cv2.imwrite(os.path.join(save_fpath, f'mask_{idx}.png'), mask * 255)
        
        # 保存NOCS坐标
        xyz = 2 * (data["nocs"][0][..., :3] - 0.5)
        np.save(os.path.join(save_fpath, f'xyz_{idx}.npy'), xyz.astype(np.float16))
    
    def render_templates(self,
                        cad_path: str,
                        output_dir: str,
                        normalize: bool = True,
                        colorize: bool = False,
                        base_color: float = 0.05) -> List[str]:
        """
        渲染CAD模型模板
        
        Args:
            cad_path: CAD模型文件路径
            output_dir: 输出目录
            normalize: 是否归一化模型尺寸
            colorize: 是否为模型着色
            base_color: 基础颜色值
            
        Returns:
            List[str]: 生成的文件路径列表
        """
        # 计算缩放因子
        scale = self.get_norm_info(cad_path) if normalize else 1.0
        
        generated_files = []
        
        for idx, cam_pose in enumerate(self.cam_poses):
            # 清空场景
            bproc.clean_up()
            
            # 加载和设置模型
            obj = bproc.loader.load_obj(cad_path)[0]
            obj.set_scale([scale, scale, scale])
            obj.set_cp("category_id", 1)
            
            # 设置材质
            if colorize:
                self._setup_material(obj, base_color)
            
            # 设置相机和光照
            self._setup_camera(cam_pose)
            self._setup_lighting(cam_pose)
            
            # 渲染
            bproc.renderer.set_max_amount_of_samples(self.max_samples)
            data = bproc.renderer.render()
            data.update(bproc.renderer.render_nocs())
            
            # 保存输出
            self._save_outputs(data, idx, output_dir)
            
            # 记录生成的文件
            template_dir = os.path.join(output_dir, "templates")
            generated_files.extend([
                os.path.join(template_dir, f'rgb_{idx}.png'),
                os.path.join(template_dir, f'mask_{idx}.png'),
                os.path.join(template_dir, f'xyz_{idx}.npy')
            ])
        
        return generated_files


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='CAD模型模板渲染器')
    parser.add_argument('--cad_path', required=True, help="CAD模型文件路径")
    parser.add_argument('--output_dir', required=True, help="输出目录路径")
    parser.add_argument('--normalize', default=True, type=bool, help="是否归一化模型")
    parser.add_argument('--colorize', default=False, type=bool, help="是否为模型着色")
    parser.add_argument('--base_color', default=0.05, type=float, help="基础颜色值")
    
    args = parser.parse_args()
    
    # 创建渲染器并执行渲染
    renderer = CADTemplateRenderer()
    renderer.render_templates(
        cad_path=args.cad_path,
        output_dir=args.output_dir,
        normalize=args.normalize,
        colorize=args.colorize,
        base_color=args.base_color
    )


if __name__ == "__main__":
    main()
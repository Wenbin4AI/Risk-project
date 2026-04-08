#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch CUDA 立方体投影生成器
基于原始Numpy版本重写，使用PyTorch和CUDA加速计算
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import time

class CubemapGeneratorTorch:
    """
    PyTorch CUDA 立方体投影生成器
    从全景图（等距柱状投影）生成六面立方体投影图，使用GPU加速
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化立方体投影生成器
        
        Args:
            device: 计算设备 ('cuda', 'cpu', 或 None 自动检测)
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置计算设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 定义立方体六个面的参数 (theta, phi, name) - 与Numpy版本一致
        self.cube_faces = {
            'front':  {'theta': 0,    'phi': 0,   'name': '前'},
            'back':   {'theta': 180,  'phi': 0,   'name': '后'},
            'left':   {'theta': -90,  'phi': 0,   'name': '左'},
            'right':  {'theta': 90,   'phi': 0,   'name': '右'},
            'top':    {'theta': 0,    'phi': 90,  'name': '上'},
            'bottom': {'theta': 0,    'phi': -90, 'name': '下'}
        }
    
    def _create_rotation_matrix(self, theta_rad: float, phi_rad: float) -> torch.Tensor:
        """
        创建旋转矩阵，精确复制Numpy版本的逻辑
        
        Args:
            theta_rad: 水平旋转角度（弧度）
            phi_rad: 垂直旋转角度（弧度）
            
        Returns:
            旋转矩阵 (3x3)
        """
        # theta旋转矩阵 (绕X轴旋转)
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        theta_rot_mat = torch.tensor([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ], dtype=torch.float32, device=self.device)
        
        # phi旋转矩阵 (绕旋转轴旋转) - Rodrigues' rotation formula
        axis_x = 0
        axis_y = math.cos(theta_rad)
        axis_z = math.sin(theta_rad)
        
        cos_phi = math.cos(phi_rad)
        sin_phi = -math.sin(phi_rad)
        
        phi_rot_mat = torch.tensor([
            [cos_phi + axis_x**2 * (1 - cos_phi),
             axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi,
             axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi],
            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi,
             cos_phi + axis_y**2 * (1 - cos_phi),
             axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi],
            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi,
             axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi,
             cos_phi + axis_z**2 * (1 - cos_phi)]
        ], dtype=torch.float32, device=self.device)
        
        # 组合旋转矩阵
        return phi_rot_mat @ theta_rot_mat
    
    def crop_panorama_face_torch(
        self, 
        img_tensor: torch.Tensor, 
        theta: float = 0.0, 
        phi: float = 0.0, 
        res_x: int = 1024, 
        res_y: int = 1024, 
        fov: float = 90.0
    ) -> torch.Tensor:
        """
        使用PyTorch从全景图裁剪指定视角的图像
        
        Args:
            img_tensor: 输入的全景图像张量 [C, H, W]
            theta: 水平旋转角度 (度)
            phi: 垂直旋转角度 (度)
            res_x: 输出图像宽度
            res_y: 输出图像高度
            fov: 视场角 (度)
            
        Returns:
            裁剪后的图像张量 [C, res_y, res_x]
        """
        
        img_height, img_width = img_tensor.shape[1], img_tensor.shape[2]
        
        # 1. 转换角度为弧度
        theta_rad = math.radians(theta)
        phi_rad = math.radians(phi)
        
        # 2. 计算FOV参数
        fov_x = fov
        aspect_ratio = res_y / res_x
        half_len_x = math.tan(math.radians(fov_x) / 2)
        half_len_y = aspect_ratio * half_len_x
        
        pixel_len_x = 2 * half_len_x / res_x
        pixel_len_y = 2 * half_len_y / res_y
        
        # 3. 创建网格坐标 - 精确复制Numpy版本行为
        x_coords = torch.arange(res_x, dtype=torch.float32, device=self.device)
        y_coords = torch.arange(res_y, dtype=torch.float32, device=self.device)
        
        map_x = x_coords.unsqueeze(1).repeat(1, res_y)
        map_y = y_coords.unsqueeze(0).repeat(res_x, 1)

        # 4. 转换到3D空间坐标
        map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
        map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
        map_z = torch.ones_like(map_x) * -1
        
        # 5. 重组为3D向量 [3, W*H]
        coords_3d = torch.stack([map_x.flatten(), map_y.flatten(), map_z.flatten()], dim=0)
        
        # 6. 应用旋转变换
        rotation_matrix = self._create_rotation_matrix(theta_rad, phi_rad)
        rotated_coords = rotation_matrix @ coords_3d
        
        # 7. 归一化到单位球面
        vec_len = torch.norm(rotated_coords, dim=0, keepdim=True)
        rotated_coords = rotated_coords / vec_len
        
        # 8. 转换为球面坐标
        cur_phi = torch.asin(torch.clamp(rotated_coords[0, :], -1, 1))
        cur_theta = torch.atan2(rotated_coords[1, :], -rotated_coords[2, :])
        
        # 9. 映射到全景图像坐标
        map_src_y = (cur_phi + math.pi/2) / math.pi * img_height
        map_src_x = (cur_theta % (2 * math.pi)) / (2 * math.pi) * img_width
        
        # 10. 重塑为2D数组
        map_src_y = map_src_y.reshape(res_x, res_y)
        map_src_x = map_src_x.reshape(res_x, res_y)
        
        # 11. 为 F.grid_sample 准备坐标 - **核心修正**
        # 修正镜像: 对一个坐标轴进行翻转。我们翻转水平方向的x坐标。
        # 原始 remap 使用 map_y 作为x坐标, map_x 作为y坐标。
        # grid_sample 使用 grid[...,0] 作为x, grid[...,1] 作为y。
        norm_x = (2.0 * map_src_x / img_width) - 1.0
        norm_y = (2.0 * map_src_y / img_height) - 1.0
        
        # 12. 创建采样网格 - **核心修正**
        # 修正旋转: 交换 norm_x 和 norm_y 的位置
        # 同时，为了正确对齐，我们不再需要对坐标进行转置 (.T)
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
        
        # 13. 使用grid_sample进行重映射
        img_batch = img_tensor.unsqueeze(0)  # [1, C, H, W]
        result = F.grid_sample(
            img_batch, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        
        return result.squeeze(0)  # [C, H, W]
    
    def generate_cubemap_batch(
        self,
        panorama_image: Union[str, np.ndarray, torch.Tensor],
        output_dir: str = "./cubemap_output",
        resolution: int = 1024,
        fov: float = 90.0,
        image_format: str = "jpg",
        prefix: str = "face",
        faces: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        批量生成立方体投影的六个面（GPU加速）
        
        Args:
            panorama_image: 全景图像路径、numpy数组或torch张量
            output_dir: 输出目录
            resolution: 每个面的分辨率 (正方形)
            fov: 视场角，通常为90度
            image_format: 输出格式 ("jpg", "png")
            prefix: 文件名前缀
            faces: 要生成的面列表，None表示全部六个面
            
        Returns:
            包含各面文件路径的字典
        """
        
        start_time = time.time()
        
        # 加载和预处理图像
        if isinstance(panorama_image, str):
            img = cv2.imread(panorama_image)
            if img is None:
                raise FileNotFoundError(f"无法读取图像文件: {panorama_image}")
            input_name = Path(panorama_image).stem
            # 转换为RGB并转换为张量
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).float().to(self.device)
        elif isinstance(panorama_image, np.ndarray):
            input_name = "panorama"
            # 假设输入是BGR格式
            img_rgb = cv2.cvtColor(panorama_image, cv2.COLOR_BGR2RGB)
            print(type(img_rgb))
            img_rgb = np.array(img_rgb)
            img_tensor = torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).float().to(self.device)
        elif isinstance(panorama_image, torch.Tensor):
            input_name = "panorama"
            img_tensor = panorama_image.to(self.device)
        else:
            raise ValueError("不支持的图像格式")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 确定要生成的面
        target_faces = faces if faces is not None else list(self.cube_faces.keys())
        
        self.logger.info(f"开始生成立方体投影图，输入图像尺寸: {img_tensor.shape[2]}x{img_tensor.shape[1]}")
        self.logger.info(f"输出分辨率: {resolution}x{resolution}, FOV: {fov}度")
        self.logger.info(f"目标面: {target_faces}")
        
        result_paths = {}
        
        with torch.no_grad():
            for face_key in target_faces:
                if face_key == "top" or face_key == "bottom":
                    break
                if face_key not in self.cube_faces:
                    self.logger.warning(f"跳过无效面: {face_key}")
                    continue
                
                face_params = self.cube_faces[face_key]
                self.logger.info(f"正在生成 {face_params['name']} 面 ({face_key})...")
                
                # 生成面图像
                face_tensor = self.crop_panorama_face_torch(
                    img_tensor=img_tensor,
                    theta=face_params['theta'],
                    phi=face_params['phi'],
                    res_x=resolution,
                    res_y=resolution,
                    fov=fov
                )
                
                # 转换回numpy并保存
                face_img_rgb = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                face_img_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
                
                # 生成文件名
                # output_filename = f"{prefix}_{face_key}_{input_name}.{image_format}"
                output_filename = f"{prefix}_{face_key}.{image_format}"
                output_filepath = output_path / output_filename
                
                # 保存图像
                success = cv2.imwrite(str(output_filepath), face_img_bgr)
                if success:
                    result_paths[face_key] = str(output_filepath)
                    self.logger.info(f"✅ {face_params['name']} 面保存至: {output_filepath}")
                else:
                    self.logger.error(f"❌ 保存 {face_params['name']} 面失败: {output_filepath}")
        
        total_time = time.time() - start_time
        self.logger.info(f"总处理时间: {total_time:.2f}秒")
        
        return result_paths
        # return panorama_image


def create_cubemap_generator_torch(device: Optional[str] = None) -> CubemapGeneratorTorch:
    """
    创建PyTorch立方体投影生成器实例
    
    Args:
        device: 计算设备
        
    Returns:
        CubemapGeneratorTorch实例
    """
    return CubemapGeneratorTorch(device=device)


if __name__ == "__main__":
    # 简单测试，确保代码可以运行
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("PyTorch CUDA 立方体投影生成器")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
    
    # 创建一个虚拟的全景图用于测试
    pano_img = np.zeros((1024, 2048, 3), dtype=np.uint8)
    pano_img[0:100, 0:100] = [255, 0, 0] # Blue square
    pano_img[-100:, -100:] = [0, 255, 0] # Green square
    
    print("\n创建一个生成器实例...")
    generator = create_cubemap_generator_torch()
    
    print("\n开始生成测试立方体贴图...")
    generator.generate_cubemap_batch(
        panorama_image=pano_img,
        output_dir="/home/ubuntu/glm/picture",
        resolution=2048
    )
    print("\n测试完成. 请检查 'test_output_torch' 目录.")

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class SingleMoleculeDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=128, 
                 transform=None, noise_level=0.1):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        self.noise_level = noise_level
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成参考图像（原始单分子分布）
        ref_image = self._generate_single_molecule_image()
        
        # 随机生成变换参数
        tx = np.random.uniform(-10, 10)  # 平移x
        ty = np.random.uniform(-10, 10)  # 平移y
        angle = np.random.uniform(-15, 15)  # 旋转角度
        scale = np.random.uniform(0.9, 1.1)  # 缩放因子
        
        # 应用变换生成目标图像
        target_image = self._apply_transform(ref_image, tx, ty, angle, scale)
        
        # 添加噪声
        ref_image = self._add_noise(ref_image)
        target_image = self._add_noise(target_image)
        
        # 转换为PyTorch张量
        ref_tensor = torch.tensor(ref_image, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0)
        
        # 归一化参数（帮助模型学习）
        params = torch.tensor([
            tx / 10,  # 归一化到[-1,1]
            ty / 10,
            angle / 15,
            (scale - 1.0) / 0.1
        ], dtype=torch.float32)
        
        return {
            'reference': ref_tensor,
            'target': target_tensor,
            'params': params,
            'metadata': {
                'tx': tx, 'ty': ty, 'angle': angle, 'scale': scale
            }
        }
    
    def _generate_single_molecule_image(self):
        # 您原有的单分子生成逻辑
        # ...
        return image
    
    def _apply_transform(self, image, tx, ty, angle, scale):
        # 应用变换逻辑
        # ...
        return transformed_image
    
    def _add_noise(self, image):
        # 添加噪声逻辑
        # ...
        return noisy_image
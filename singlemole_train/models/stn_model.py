import torch
import torch.nn as nn
import torch.nn.functional as F

class RegistrationNetwork(nn.Module):
    def __init__(self, img_size=128):
        super(RegistrationNetwork, self).__init__()
        self.img_size = img_size
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        # 计算特征图大小
        feature_size = 64 * (img_size // 8) * (img_size // 8)
        
        # 回归网络预测变换参数
        self.regressor = nn.Sequential(
            nn.Linear(feature_size * 2, 256),  # *2 因为拼接了两个图像的特征
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 4)  # 输出 tx, ty, angle, scale
        )
    
    def forward(self, ref_img, target_img):
        # 提取特征
        ref_features = self.feature_extractor(ref_img)
        target_features = self.feature_extractor(target_img)
        
        # 展平特征
        ref_features = ref_features.view(ref_features.size(0), -1)
        target_features = target_features.view(target_features.size(0), -1)
        
        # 拼接特征
        concat_features = torch.cat([ref_features, target_features], dim=1)
        
        # 预测变换参数
        params = self.regressor(concat_features)
        
        return params

def params_to_matrix(params, img_size):
    """将网络预测的参数转换为仿射变换矩阵"""
    batch_size = params.size(0)
    
    # 反归一化参数
    tx = params[:, 0] * 10
    ty = params[:, 1] * 10
    angle = params[:, 2] * 15 * np.pi / 180  # 转为弧度
    scale = params[:, 3] * 0.1 + 1.0
    
    # 构建旋转矩阵
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    
    # 构建仿射变换矩阵 [2x3]
    matrix = torch.zeros(batch_size, 2, 3, device=params.device)
    matrix[:, 0, 0] = scale * cos
    matrix[:, 0, 1] = -scale * sin
    matrix[:, 0, 2] = tx
    matrix[:, 1, 0] = scale * sin
    matrix[:, 1, 1] = scale * cos
    matrix[:, 1, 2] = ty
    
    return matrix
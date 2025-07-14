import torch
import numpy as np
import matplotlib.pyplot as plt
from model import RegistrationNetwork, params_to_matrix
from codefile.singlemole_train.data.data_generator import SingleMoleculeDataset

def visualize_registration(model, dataset, idx=0, device='cuda'):
    model.to(device)
    model.eval()
    
    # 获取样本
    sample = dataset[idx]
    ref = sample['reference'].unsqueeze(0).to(device)
    target = sample['target'].unsqueeze(0).to(device)
    true_params = sample['params'].unsqueeze(0).to(device)
    
    # 预测变换参数
    with torch.no_grad():
        pred_params = model(ref, target)
    
    # 将参数转换为变换矩阵
    pred_matrix = params_to_matrix(pred_params, ref.size(2))
    
    # 使用STN对目标图像进行变换
    grid = torch.nn.functional.affine_grid(pred_matrix, target.size(), align_corners=False)
    warped = torch.nn.functional.grid_sample(target, grid, align_corners=False)
    
    # 转换为numpy数组用于可视化
    ref_np = ref.cpu().numpy()[0, 0]
    target_np = target.cpu().numpy()[0, 0]
    warped_np = warped.cpu().numpy()[0, 0]
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(ref_np, cmap='gray')
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_np, cmap='gray')
    axes[0, 1].set_title('Target Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(warped_np, cmap='gray')
    axes[1, 0].set_title('Registered Image (Predicted)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ref_np, cmap='gray', alpha=0.5)
    axes[1, 1].imshow(warped_np, cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Overlay (Reference + Registered)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 打印真实和预测的参数
    print(f"True Parameters: tx={true_params[0,0]:.4f}, ty={true_params[0,1]:.4f}, angle={true_params[0,2]:.4f}, scale={true_params[0,3]:.4f}")
    print(f"Pred Parameters: tx={pred_params[0,0]:.4f}, ty={pred_params[0,1]:.4f}, angle={pred_params[0,2]:.4f}, scale={pred_params[0,3]:.4f}")

if __name__ == '__main__':
    # 加载模型
    model = RegistrationNetwork()
    model.load_state_dict(torch.load('best_registration_model.pth'))
    
    # 创建数据集
    dataset = SingleMoleculeDataset(num_samples=100)
    
    # 可视化配准结果
    visualize_registration(model, dataset, idx=0)    
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.data_generator import SingleMoleculeDataset
from models.stn_model import RegistrationNetwork, params_to_matrix
from utils.metrics import calculate_registration_error

def train(args):
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据集和数据加载器
    train_dataset = SingleMoleculeDataset(
        num_samples=args.num_train_samples,
        img_size=args.img_size,
        noise_level=args.noise_level
    )
    val_dataset = SingleMoleculeDataset(
        num_samples=args.num_val_samples,
        img_size=args.img_size,
        noise_level=args.noise_level
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = RegistrationNetwork(img_size=args.img_size)
    model.to(args.device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard日志
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            ref_images = batch['reference'].to(args.device)
            target_images = batch['target'].to(args.device)
            true_params = batch['params'].to(args.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            pred_params = model(ref_images, target_images)
            loss = criterion(pred_params, true_params)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_reg_error = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                ref_images = batch['reference'].to(args.device)
                target_images = batch['target'].to(args.device)
                true_params = batch['params'].to(args.device)
                
                # 前向传播
                pred_params = model(ref_images, target_images)
                loss = criterion(pred_params, true_params)
                
                val_loss += loss.item()
                
                # 计算配准误差（以像素为单位）
                batch_size = ref_images.size(0)
                for i in range(batch_size):
                    true_matrix = params_to_matrix(true_params[i:i+1], args.img_size)
                    pred_matrix = params_to_matrix(pred_params[i:i+1], args.img_size)
                    
                    reg_error = calculate_registration_error(
                        ref_images[i:i+1], target_images[i:i+1],
                        true_matrix, pred_matrix, args.device
                    )
                    val_reg_error += reg_error
        
        # 计算平均损失和误差
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_reg_error /= len(val_dataset)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/registration_error', val_reg_error, epoch)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Registration Error: {val_reg_error:.4f} pixels')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Model saved at epoch {epoch+1}')
    
    writer.close()
    print('Training completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Molecule Image Registration Training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_train_samples', type=int, default=10000)
    parser.add_argument('--num_val_samples', type=int, default=1000)
    parser.add_argument('--noise_level', type=float, default=0.1)
    
    args = parser.parse_args()
    train(args)
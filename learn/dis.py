import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def train(rank, world_size, args):
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    
    # 创建模型并移动到 GPU
    model = SimpleModel().to(device)
    
    # 包装模型
    model = DDP(model, device_ids=[rank])
    
    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item()}")

    # 清理分布式环境
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()

    world_size = 2  # 使用两个 GPU
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
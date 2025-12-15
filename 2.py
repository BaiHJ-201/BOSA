import torch
import torch.ao.quantization
import torch.ao.quantization.fx._lower_to_native_backend
if __name__ == '__main__':
    # 先检查CUDA是否可用，避免无GPU时报错
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输入（直接to(device)，兼容GPU/CPU）
    x = torch.rand(size=(512, 3, 224, 224), requires_grad=False).to(device)
    
    # 加载模型时指定map_location，避免设备不匹配问题
    model = torch.load("/root/WZR/TRIBE/mobilenet_v2_CIFAR10_compressed.pth", map_location=device)
    model = model.to(device)
    
    # 量化模型需切换到eval模式（否则可能报错）
    model.eval()
    
    # 禁用梯度计算，节省显存+避免量化模型梯度报错
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
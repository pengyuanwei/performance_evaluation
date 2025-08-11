import torch
import time

def test_transfer_speed(tensor_size_mb=100):
    # tensor大小，单位MB，转换为元素数量（float32，每个元素4字节）
    num_elements = tensor_size_mb * 1024 * 1024 // 4
    
    # 在CPU创建一个float32张量
    cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
    
    # 确保GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("当前没有GPU，无法测试GPU传输速度")
        return
    
    # 测量Host to Device传输时间
    torch.cuda.synchronize()
    start = time.time()
    gpu_tensor = cpu_tensor.to(device)
    torch.cuda.synchronize()
    end = time.time()
    h2d_time = end - start
    
    # 测量Device to Host传输时间
    torch.cuda.synchronize()
    start = time.time()
    cpu_tensor_back = gpu_tensor.to('cpu')
    torch.cuda.synchronize()
    end = time.time()
    d2h_time = end - start
    
    # 计算带宽 GB/s
    size_gb = tensor_size_mb / 1024
    h2d_bandwidth = size_gb / h2d_time
    d2h_bandwidth = size_gb / d2h_time
    
    print(f"Tensor size: {tensor_size_mb} MB")
    print(f"Host to Device bandwidth: {h2d_bandwidth:.2f} GB/s")
    print(f"Device to Host bandwidth: {d2h_bandwidth:.2f} GB/s")

if __name__ == '__main__':
    test_transfer_speed(4000)  # 测试100MB数据传输速度
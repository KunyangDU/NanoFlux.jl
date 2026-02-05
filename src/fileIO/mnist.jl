"""
数据加载器构建函数
负责下载、维度重塑 (增加通道)、类型转换和分批
"""
function mnist_loader(batch_size::Int)
    
    # 1. 加载原始数据
    # train_x_raw: (28, 28, 60000) :: Matrix{Gray{N0f8}} 或 Float32
    # train_y_raw: (60000,) :: Int64
    train_x_raw, train_y_raw = MNIST(split=:train)[:]
    
    # 2. 预处理 X (Features)
    # 原始形状: (Width, Height, Batch) -> (28, 28, 60000)
    # 目标形状: (Channel, Height, Width, Batch) -> (1, 28, 28, 60000)
    # -----------------------------------------------------------
    # [关键修改]
    # 我们不再 flatten 成 784。
    # 而是插入 Channel=1 的维度，适配 SpatialTensor{2}。
    # -----------------------------------------------------------
    x_reshaped = reshape(train_x_raw, 1, 28, 28, :)
    x_data = Float32.(x_reshaped) # 强烈建议用 Float32
    
    # 3. 预处理 Y (Labels)
    # 目标形状: (Classes, Batch) -> (10, 60000)
    # 使用 One-Hot 编码
    y_oh = onehotbatch(train_y_raw, 0:9)
    y_data = Float32.(y_oh) # 标签也转为 Float32，方便计算 Loss
    
    # 4. 构建 DataLoader
    # MLUtils 会自动沿着最后一个维度 (dim=4 for x, dim=2 for y) 切片
    # 每次迭代返回: 
    #   x_batch: (1, 28, 28, batch_size)
    #   y_batch: (10, batch_size)
    loader = DataLoader((x_data, y_data), batchsize=batch_size, shuffle=true)
    
    return loader
end
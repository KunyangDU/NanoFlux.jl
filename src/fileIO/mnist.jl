"""
数据加载器构建函数
负责下载、展平、类型转换和分批
"""
function mnist_loader(batch_size::Int)
    
    # 1. 加载原始数据
    # train_x_raw: (28, 28, 60000) Float32
    # train_y_raw: (60000,) Int64
    train_x_raw, train_y_raw = MNIST(split=:train)[:]
    
    # 2. 预处理 X (Features)
    # 目标形状: (Features, Total_Samples) -> (784, 60000)
    # 这一步直接把空间维度展平，列方向是样本
    x_flat = reshape(train_x_raw, 28 * 28, :)
    x_data = Float64.(x_flat) # 转为 Float64 配合 TensorKit
    
    # 3. 预处理 Y (Labels)
    # 目标形状: (Classes, Total_Samples) -> (10, 60000)
    y_oh = onehotbatch(train_y_raw, 0:9)
    y_data = Float64.(y_oh)
    
    # 4. 构建 DataLoader
    # MLUtils 会自动沿着最后一个维度 (dim 2) 切片
    # 每次迭代返回: x_batch (784, 64), y_batch (10, 64)
    loader = DataLoader((x_data, y_data), batchsize=batch_size, shuffle=true)
    
    return loader
end
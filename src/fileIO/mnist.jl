"""
数据加载器构建函数
负责下载、维度重塑 (增加通道)、类型转换和分批
"""
function mnist_loader(batch_size::Int)
    train_x_raw, train_y_raw = MNIST(split=:train)[:]
    x_reshaped = reshape(train_x_raw, 28, 28, 1, :)
    x_data = Float32.(x_reshaped)
    y_oh = onehotbatch(train_y_raw, 0:9)
    y_data = Float32.(y_oh)
    loader = DataLoader((x_data, y_data), batchsize=batch_size, shuffle=true)
    return loader
end
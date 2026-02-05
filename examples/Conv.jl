include("../src/NanoFlux.jl")
# 单独训练一个卷积层

function loss(model::Conv, x, y)
    y_pred = model(x) # 输出是 SpatialTensor
    diff = y_pred.data .- y
    L = sum(abs2, diff) / length(y)
    return L
end

accuracy(model::Conv, x, y) = 0.0

function test_single_conv()
    # println(b"\n🧪 TEST 2: Training a Single Conv Layer (Regression)")
    println("="^60)
    println("ℹ️  Note: Using MSE Loss specifically for Conv layer testing.")

    # 1. 构造数据 (拟合输入图片 -> 输出 Feature Map)
    # 输入: 1通道, 10x10 图片
    # 目标: 4通道, 对应卷积后的尺寸
    # Conv 配置: 2D, 1->4, Kernel=3, Stride=1 => OutSize = 10-3+1 = 8
    
    H, W = 10, 10
    C_in, C_out = 1, 4
    K = 3
    Out_H, Out_W = 8, 8
    
    # 构造随机输入
    X_raw = randn(Float32, 1, H, W, 100) # (C, H, W, N)
    
    # 构造随机"目标" (这就好比让卷积层去学习某种特定的滤波效果)
    Y_target = randn(Float32, C_out, Out_H, Out_W, 100) 
    
    loader = DataLoader((X_raw, Y_target), batchsize=10, shuffle=true)

    # 2. 实例化单独的 Conv 层
    model = Conv(2, C_in, C_out, K; act=identity)

    # 3. 训练
    # 这一步会自动调用上面定义的 specialized loss(::Conv, ...)
    algo = SimpleAlgorithm(epochs=10, learning_rate=1e-3, show_times=5)
    
    train!(model, loader, algo)
    
    println("\n",bg"✅ Single Conv Layer Test Passed!")
end


test_single_conv()


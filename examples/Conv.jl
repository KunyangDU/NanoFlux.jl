include("../src/NanoFlux.jl")

# 显式特化针对 Conv 类型的 loss，覆盖 train.jl 中的默认逻辑
function loss(model::Conv, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsType)
    y_pred = model(x, ps) # ⚠️ 必须传入 ps
    diff = y_pred.data .- y
    L = sum(abs2, diff) / length(y)
    return L
end

# Accuracy 对回归任务无意义，返回 0 以避免报错
accuracy(model::Conv, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsType) = 0.0

function test_single_conv()
    println("\n🧪 TEST 2: Training a Single Conv Layer (Regression)")
    println("="^60)
    println("ℹ️  Note: Using MSE Loss specifically for Conv layer testing.")

    H, W = 10, 10
    C_in, C_out = 1, 4
    K = 3
    # Conv 配置: Kernel=3, Stride=1, Dilation=1 => OutSize = 10 - 3 + 1 = 8
    Out_H, Out_W = 8, 8
    
    # (Channel, H, W, Batch)
    X_raw = randn(Float32, 1, H, W, 100) 
    Y_target = randn(Float32, C_out, Out_H, Out_W, 100) 
    
    loader = DataLoader((X_raw, Y_target), batchsize=10, shuffle=true)

    model = Conv(2, C_in, C_out, K; act=identity)

    opt = Adam(learning_rate=1e-2) 
    config = TrainerConfig(epochs=20, show_times=5) # 增加 epochs 确保拟合
    
    train!(model, loader, opt, config)
    
    println("\n", bg"✅ Single Conv Layer Test Passed!")
end

test_single_conv()
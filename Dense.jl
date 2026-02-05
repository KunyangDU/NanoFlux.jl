include("src/NanoFlux.jl")
# 单独训练一个全连接层

function test_single_dense()
    println("\n🧪 TEST 1: Training a Single Dense Layer (Classification)")
    println("="^60)

    # 1. 构造数据
    # 输入: 64维向量, 1000个样本
    # 输出: 10类 (One-Hot)
    InputDim = 64
    OutputDim = 10
    BatchSize = 32
    
    X = randn(Float32, InputDim, 1000)
    Y_labels = rand(1:OutputDim, 1000)
    
    # 手动 One-Hot (为了不依赖外部库)
    Y = zeros(Float32, OutputDim, 1000)
    for (i, label) in enumerate(Y_labels)
        Y[label, i] = 1.0f0
    end
    
    loader = DataLoader((X, Y), batchsize=BatchSize, shuffle=true)

    # 2. 实例化单独的层
    # 注意：这里不需要 Sequential，直接用 Dense
    model = Dense(InputDim, OutputDim, identity) # 最后一层通常不用激活
    
    # println("Layer Info: $(model)")
    println("Params W: $(size(model.W))")

    # 3. 设置算法
    algo = SimpleAlgorithm(epochs=5, learning_rate=1e-2, show_times=10)

    # 4. 开始训练
    # train! 会自动识别它是一个 AbstractModule
    train!(model, loader, algo)
    
    println("\n",bg"✅ Single Dense Layer Test Passed!")
end

test_single_dense()
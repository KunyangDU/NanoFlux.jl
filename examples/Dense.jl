include("../src/NanoFlux.jl")

function test_single_dense()
    println("\n🧪 TEST 1: Training a Single Dense Layer (Classification)")
    println("="^60)

    InputDim = 64
    OutputDim = 10
    BatchSize = 32
    
    X = randn(Float32, InputDim, 1000)
    Y_labels = rand(1:OutputDim, 1000)

    Y = zeros(Float32, OutputDim, 1000)
    for (i, label) in enumerate(Y_labels)
        Y[label, i] = 1.0f0
    end
    
    loader = DataLoader((X, Y), batchsize=BatchSize, shuffle=true)

    model = Dense(InputDim, OutputDim, identity) 
    
    temp_ps = initialize(model)
    println("Layer Config: $(model)")
    println("Params W Shape: $(size(temp_ps.W))") # 从 ps 中获取

    opt = SGD(learning_rate=1e-1, momentum=0.9)
    config = TrainerConfig(epochs=10, show_times=10)

    train!(model, initialize(model), loader, opt, config)
    
    println("\n", bg"✅ Single Dense Layer Test Passed!")
end

test_single_dense()
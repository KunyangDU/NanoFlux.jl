include("../src/NanoFlux.jl")

function load_cifar10(; batch_size::Int=32, split::Symbol=:train, normalize::Bool=true)
    imgs, labels = CIFAR10(split=split)[:]
    x_data = Float32.(imgs)
    normalize && (@. x_data = 2f0 * x_data - 1f0)
    dataset = SpatialDataset(x_data, labels; num_classes=10, add_channel_dim=false)
    loader = DataLoader(dataset, batchsize=batch_size, shuffle=(split == :train))
    return loader
end


BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 5
CUT_STEP = Inf
BASE_CH = 32  
TIME_DIM = BASE_CH * 4 
TIME_STEP = 400

# 保持层数不变 (RealUNet 结构不变)，但大幅减小每层的“宽度”
model = UNet(
    TimeEmbedding(TIME_DIM),
    
    # Input: 3 -> 32
    Conv(2, 3, BASE_CH, 3, stride=1, pad=1),
    
    # --- Down 1 (32x32 -> 16x16) ---
    # 输入 32 -> 输出 64 (通道翻倍)
    ResNetBlock(BASE_CH, BASE_CH*2, TIME_DIM), 
    ResNetBlock(BASE_CH*2, BASE_CH*2, TIME_DIM),
    Conv(2, BASE_CH*2, BASE_CH*2, 3, stride=2, pad=1), # Downsample
    
    # --- Down 2 (16x16 -> 8x8) ---
    # 输入 64 -> 输出 128 (通道翻倍)
    ResNetBlock(BASE_CH*2, BASE_CH*4, TIME_DIM),
    ResNetBlock(BASE_CH*4, BASE_CH*4, TIME_DIM),
    SpatialAttention(BASE_CH*4), # 在 128 通道上做 Attention
    Conv(2, BASE_CH*4, BASE_CH*4, 3, stride=2, pad=1), # Downsample
    
    # --- Middle (8x8) ---
    # 保持 128 通道
    ResNetBlock(BASE_CH*4, BASE_CH*4, TIME_DIM),
    SpatialAttention(BASE_CH*4),
    ResNetBlock(BASE_CH*4, BASE_CH*4, TIME_DIM),
    
    # --- Up 2 (8x8 -> 16x16) ---
    # Upsample
    Conv(2, BASE_CH*4, BASE_CH*4, 3, stride=1, pad=1),
    # 拼接后输入: 128(本身) + 128(Skip) = 256 -> 输出降回 64
    ResNetBlock(BASE_CH*4 + BASE_CH*4, BASE_CH*2, TIME_DIM),
    ResNetBlock(BASE_CH*2, BASE_CH*2, TIME_DIM),
    
    # --- Up 1 (16x16 -> 32x32) ---
    # Upsample
    Conv(2, BASE_CH*2, BASE_CH*2, 3, stride=1, pad=1),
    # 拼接后输入: 64(本身) + 64(Skip) = 128 -> 输出降回 32
    ResNetBlock(BASE_CH*2 + BASE_CH*2, BASE_CH, TIME_DIM),
    ResNetBlock(BASE_CH, BASE_CH, TIME_DIM),
    
    # --- Out ---
    Sequential(
        # GroupNorm 的 groups 必须能整除通道数
        # 32 通道正好可以用 32 组 (Instance Norm) 或者 8 组
        GroupNorm(32, BASE_CH), 
        Identity(silu),
        Conv(2, BASE_CH, 3, 3, stride=1, pad=1) # Output: 3 (RGB)
    )
)
@save "examples/data/ddpm_model_CIFAR10_T$(TIME_STEP)_CH$(BASE_CH)_B$(BATCH_SIZE)_LR$(LEARNING_RATE)_EP$(EPOCHS)_Cut$(CUT_STEP).jld2" model

# ps = initialize(model)
@load "examples/data/ddpm_ps_CIFAR10_T$(TIME_STEP)_CH$(BASE_CH)_B$(BATCH_SIZE)_LR$(LEARNING_RATE)_EP$(EPOCHS)_Cut$(CUT_STEP).jld2" ps
LEARNING_RATE = 1e-5
opt = Adam(
    learning_rate = LEARNING_RATE
)

config = TrainerConfig(
    epochs = EPOCHS, 
    show_times = 10, 
    cut_step=CUT_STEP,
    config = DiffusionProcess(TIME_STEP)
)

train_loader = load_cifar10(batch_size=BATCH_SIZE, split=:train)

ps,history = train!(model, ps, train_loader, opt, config)

@save "examples/data/ddpm_ps_CIFAR10_T$(TIME_STEP)_CH$(BASE_CH)_B$(BATCH_SIZE)_LR$(LEARNING_RATE)_EP$(EPOCHS)_Cut$(CUT_STEP).jld2" ps
@save "examples/data/ddpm_history_CIFAR10_T$(TIME_STEP)_CH$(BASE_CH)_B$(BATCH_SIZE)_LR$(LEARNING_RATE)_EP$(EPOCHS)_Cut$(CUT_STEP).jld2" history

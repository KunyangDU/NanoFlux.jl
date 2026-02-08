function loss(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer, ::NoAlgorithm)
    y_pred = model(x, ps)
    logits = y_pred.data
    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    return -sum(y .* log.(probs .+ 1.0f-10)) / size(logits, 2)
end

function loss(model::AbstractModule, x::AbstractNanoTensor, y::Matrix{Int}, ps::ParamsContainer, ::NoAlgorithm)
    # 1. 前向传播 -> (Vocab, Seq, Batch)
    logits = model(x, ps).data 
    
    # 2. 数值稳定的 Log Softmax
    # 直接在原张量上操作，Zygote 会自动处理反向传播
    log_probs = logsoftmax(logits, dims=1) 

    # 3. 高效 Gather (取出目标位置的概率)
    # 我们不需要把所有数据 reshape 成 2D，直接计算线性索引
    
    V, S, B = size(logits)
    
    # 计算 y 中每个 label 在 log_probs 中的线性索引 (Linear Index)
    # log_probs 是 (V, S, B)，也就是列优先存储
    # 对于第 (s, b) 个位置，其目标 label 是 y[s, b]
    # 其在 log_probs 中的线性偏移量 = (b-1)*V*S + (s-1)*V + y[s,b]
    
    # 构建基础偏移量 (0, V, 2V...) 对应每个列向量的起始位置
    # 这一步利用了 Julia 的线性内存布局特性
    # 0:(S*B-1) 生成 0, 1, 2...
    # .* V 变成 0, V, 2V...
    col_offsets = (0:(S*B - 1)) .* V
    
    # y 展平后就是每个位置具体的"行号" (1-based index)
    # 最终索引 = 行号 + 列偏移
    # reshape(y, :) 变成 (N,)
    target_indices = reshape(y, :) .+ col_offsets
    
    # 4. 取值并计算平均负对数似然 (Mean NLL)
    # view 避免复制内存
    # mean 会自动处理标量除法
    return -mean(view(log_probs, target_indices))
end

# function accuracy(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer)
#     y_pred = model(x, ps)
#     logits = y_pred.data
#     pred_idx = [c[1] for c in argmax(logits, dims=1)]
#     true_idx = [c[1] for c in argmax(y, dims=1)]
#     return mean(pred_idx .== true_idx)
# end
function accuracy(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer, ::NoAlgorithm)
    y_pred = model(x, ps)
    logits = y_pred.data
    
    # 1. 获取最大值的索引 (GPU 上的 CartesianIndex 数组)
    # 此时 pred_indices 和 true_indices 依然在显存中
    pred_indices = argmax(logits, dims=1)
    true_indices = argmax(y, dims=1)
    
    # 2. 直接在 GPU 上进行广播比较
    # CartesianIndex 支持直接 == 比较，不需要提取 c[1]
    # 结果是一个 GPU 上的 BitArray (Bool)
    matches = pred_indices .== true_indices
    
    # 3. mean 支持 GPU 数组，直接返回结果
    return mean(matches)
end
function accuracy(model::AbstractModule, x::AbstractNanoTensor, y::Matrix{Int}, ps::ParamsContainer, ::NoAlgorithm)
    # 1. 前向传播
    logits = model(x, ps).data # (Vocab, Seq, Batch)

    # 2. 获取预测类别
    # argmax(logits, dims=1) 得到 (1, Seq, Batch) 的 CartesianIndex 数组
    # 我们不 dropdims，而是让 y 配合它
    pred_indices = argmax(logits, dims=1) 

    # 3. 比较
    # y 是 (Seq, Batch)，我们将其 reshape 为 (1, Seq, Batch) 以便广播
    # c[1] 取出 CartesianIndex 的第一个维度（即预测的 Token ID）
    # 这一步会在 GPU 上自动融合为一个 Kernel，非常快
    y_reshaped = reshape(y, 1, size(y)...)
    
    return mean(getindex.(pred_indices, 1) .== y_reshaped)
end

# 定义一个全局或传入的 DiffusionProcess 实例
# 为了方便，这里假设你会在 main 脚本里定义它，或者将其设为全局常量
# const DIFFUSION = DiffusionProcess(1000) 

# 重载 loss 函数
function loss(model::UNet, x::SpatialTensor, ::AbstractArray, ps::ParamsContainer, DIFFUSION::DiffusionProcess)
    # x: (W, H, C, B) - 真实图片
    # y: 在 DDPM 训练中通常被忽略（或者是条件标签，这里暂时忽略）
    
    batch_size = size(x, 4)
    device = x.data isa Array ? CPU() : GPU() # 简单的设备判断
    
    # 1. 随机采样时间步 t
    # t ~ Uniform(1, 1000)
    t = rand(1:DIFFUSION.timesteps, batch_size) 
    
    # 2. 生成随机噪声 epsilon
    noise = randn(Float32, size(x))
    if device isa GPU
        noise = move_to(GPU(), noise)
        # t 保持在 CPU 用于索引，或者索引后再移到 GPU，取决于 q_sample 实现
        # 通常建议: t 留在 CPU 做索引，取出的系数移到 GPU
    end
    
    # 3. 加噪 (Forward Process)
    # 注意：你需要确保 DIFFUSION 全局变量存在，或者将其传进来
    # 这里为了演示，假设有一个全局 DIFFUSION 对象
    x_t_data = q_sample(DIFFUSION, x.data, t, noise)
    x_t = SpatialTensor{2}(x_t_data)
    
    # 4. 模型预测噪声
    # 你的 UNet forward 签名是 (x, t, ps)
    pred_noise = model(x_t, t, ps)
    
    # 5. 计算 MSE Loss
    diff = noise .- pred_noise.data
    return mean(abs2, diff)
end

# 重载 accuracy (DDPM 不需要 accuracy，返回 0 或者 MSE)
function accuracy(model::UNet, x::SpatialTensor, y::AbstractArray, ps::ParamsContainer, DIFFUSION::DiffusionProcess)
    return 0.0f0 
end
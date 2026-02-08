# 前向加噪核心公式: q(x_t | x_0)
# x_0: (W, H, C, B)
# t: (B,)
# noise: (W, H, C, B)
q_sample(d::DiffusionProcess, x_0::AbstractArray, t::Vector{Int}, noise::AbstractArray) = reshape(d.sqrt_alpha_bar[t], 1, 1, 1, :) .* x_0 .+ reshape(d.sqrt_one_minus_alpha_bar[t], 1, 1, 1, :) .* noise

"""
    p_sample(model, ps, d, x, t_batch, t_scalar)

反向过程的单步计算: x_{t-1} ~ p_theta(x_{t-1}|x_t)
公式: x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * eps) + sigma * z
"""
function p_sample(model, ps, d, x::AbstractArray, t_batch::Vector{Int}, t::Int)
    β_t = d.beta[t]
    α_t = d.alpha[t]
    sqrt_recip_α_t = 1.0f0 / sqrt(α_t)
    
    # sqrt(1 - alpha_bar_t)
    sqrt_one_minus_α_bar_t = d.sqrt_one_minus_alpha_bar[t]
    
    # 系数: (1 - α_t) / sqrt(1 - α_bar_t)
    coeff_eps = (1.0f0 - α_t) / sqrt_one_minus_α_bar_t
    
    # 方差 σ_t (论文中通常取 sqrt(β_t))
    σ_t = sqrt(β_t)
    
    # 包装成 SpatialTensor
    x_tensor = SpatialTensor{2}(x) 
    pred_noise = model(x_tensor, t_batch, ps).data
    
    # mean = 1/sqrt(α) * (x - coeff * eps)
    mean = sqrt_recip_α_t .* (x .- coeff_eps .* pred_noise)
    
    return t > 1 ? (mean .+ σ_t .* randn(Float32, size(x))) : mean
end

"""
    sample_ddpm(model, ps, diffusion, shape; save_path=nothing)

DDPM 采样主函数 (Algorithm 2)。
- shape: (W, H, C, BatchSize), 例如 (32, 32, 3, 16)
"""
function sample(model::AbstractModule, ps::ParamsContainer, d::DiffusionProcess, shape::Tuple)
    
    # 初始化纯噪声 x_T ~ N(0, I)
    device = ps.inc.W isa Array ? CPU() : GPU() # 检测模型在哪个设备
    img = randn(Float32, shape) |> x -> move_to(device, x)
    
    println("Start Sampling from T=$(d.timesteps)...")
    
    # 反向去噪循环 (从 T 到 1)
    for t in d.timesteps:-1:1
        # 构造时间步 batch: [t, t, ..., t]
        t_batch = fill(t, shape[end])

        # 执行单步去噪
        img = p_sample(model, ps, d, img, t_batch, t)
        
        # (可选) 打印进度
        if t % 5 == 0
            print("$t ")
        end
    end
    println("\nDone!")

    # 最终处理：映射回 [0, 1] 并转为图片对象
    # DDPM 的输出通常在 [-1, 1] 之间，需要 clamp 一下防止溢出
    img_clamped = clamp.(img, -1.0f0, 1.0f0)
    img_norm = (img_clamped .+ 1.0f0) ./ 2.0f0
    final_data = img_norm |> x -> move_to(CPU(), x)
    
    return [colorview(RGB, permutedims(final_data[:,:,:,i], (3, 2,1))) for i in 1:size(final_data)[end]]
end
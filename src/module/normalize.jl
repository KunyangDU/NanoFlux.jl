
"""
    LayerNorm(features::Int; eps=1e-5)

层归一化 (Layer Normalization)。
在 Transformer 中，它对每个 Token 的特征向量进行归一化 (零均值，单位方差)。

公式: y = (x - μ) / √(σ² + ε) * γ + β

输入/输出: (Embed_Dim, Seq_Len, Batch)
"""
struct LayerNorm <: AbstractModule
    features::Int
    eps::Float32
end

LayerNorm(features::Int; eps=1e-5) = LayerNorm(features, Float32(eps))

function initialize(l::LayerNorm, rng::TaskLocalRNG = Random.default_rng())
    return (
        # γ (scale): 初始化为 1，保持原样
        gamma = ones(Float32, l.features),
        # β (bias): 初始化为 0
        beta  = zeros(Float32, l.features)
    )
end

function (l::LayerNorm)(x::SpatialTensor{D}, ps::ParamsContainer) where D
    # x.data: (Features, Seq, Batch)
    u = x.data
    μ = mean(u, dims=1)
    σ² = mean(abs2.(u .- μ), dims=1)
    x_norm = (u .- μ) ./ sqrt.(σ² .+ l.eps)
    y = x_norm .* ps.gamma .+ ps.beta
    return SpatialTensor{D}(y)
end

function Base.show(io::IO, l::LayerNorm)
    print(io, "LayerNorm($(l.features))")
end


# 放在 src/utils.jl

# 放在 src/module/normalize.jl
# GroupNorm 实现 (简化版，将 channel 分组归一化)
struct GroupNorm <: AbstractModule
    num_groups::Int
    num_channels::Int
    eps::Float32
end
GroupNorm(g::Int, c::Int) = GroupNorm(g, c, 1.0f-5)

function initialize(l::GroupNorm, ::TaskLocalRNG)
    return (weight = ones(Float32, l.num_channels), bias = zeros(Float32, l.num_channels))
end

function (gn::GroupNorm)(x::SpatialTensor{D}, ps) where D
    # x: (W, H, C, B)
    W, H, C, B = size(x)
    G = gn.num_groups
    @assert C % G == 0 "Channel must be divisible by groups"
    
    # Reshape: (W, H, C/G, G, B) -> 归一化维度: (W, H, C/G)
    x_reshaped = reshape(x.data, W, H, div(C, G), G, B)
    
    # 计算均值和方差 (dims=1,2,3)
    μ = mean(x_reshaped, dims=(1,2,3))
    σ² = var(x_reshaped, dims=(1,2,3), mean=μ, corrected=false)
    
    # 归一化
    x_norm = (x_reshaped .- μ) ./ sqrt.(σ² .+ gn.eps)
    
    # 还原形状并应用仿射变换 (Weight & Bias 需要 reshape 以广播)
    x_out = reshape(x_norm, W, H, C, B)
    w_shape = (ntuple(_->1, D)..., C, 1)
    
    return SpatialTensor{D}(x_out .* reshape(ps.weight, w_shape) .+ reshape(ps.bias, w_shape))
end
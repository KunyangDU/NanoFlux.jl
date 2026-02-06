
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
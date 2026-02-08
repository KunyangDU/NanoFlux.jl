# 一个简单的正弦位置编码 + MLP 投影
struct TimeEmbedding <: AbstractModule
    dim::Int
    mlp::Sequential
end

function TimeEmbedding(dim::Int)
    # t -> Sinusoidal -> Dense -> SiLU -> Dense
    return TimeEmbedding(dim, Sequential(
        Dense(dim, dim * 4, x->x .* sigmoid.(x)), # SiLU 近似
        Dense(dim * 4, dim)
    ))
end

initialize(m::TimeEmbedding, rng::TaskLocalRNG) = initialize(m.mlp, rng)

function (m::TimeEmbedding)(t::Vector{Int}, ps::ParamsContainer)
    # 正弦编码
    half_dim = m.dim ÷ 2
    emb_scale = log(10000f0) / (half_dim - 1)
    emb = exp.(-Float32.(0:half_dim-1) * emb_scale) # (Half,)
    
    # (Half, Batch)
    emb = reshape(emb, :, 1) .* reshape(Float32.(t)', 1, :) 
    emb = vcat(sin.(emb), cos.(emb)) # (Dim, Batch)
    
    # MLP 投影
    # 需要把 FlatTensor 包装传给 Dense，或者直接调用
    # 这里假设你的 Dense 接受 Matrix
    # NanoFlux 的 Dense 接受 FlatTensor
    return m.mlp(FlatTensor(emb), ps).data # 返回 Matrix
end
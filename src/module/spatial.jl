"""
    SpatialAttention(channels::Int; heads::Int=4, groups::Int=32)

图像自注意力模块，用于处理 SpatialTensor{2} (W, H, C, B)。
它将图像展平为序列，调用通用的 Attention，然后再还原。
"""
struct SpatialAttention <: AbstractModule
    norm::GroupNorm
    attn::Attention # 复用你重构的 Attention
end

function SpatialAttention(channels::Int; heads::Int=4, groups::Int=32)
    # 1. 图像注意力通常不需要因果遮罩 (causal=false)
    # 2. max_len: 此时 mask 为 nothing，max_len 不影响逻辑，但为了满足构造函数签名，
    #    我们可以传一个典型值 (例如 32x32=1024) 或 0。
    #    Attention 内部的 forward 会根据输入动态获取 T，所以这里传 0 是安全的。
    return SpatialAttention(
        GroupNorm(groups, channels),
        Attention(channels, heads, 0; causal=false) 
    )
end

function initialize(m::SpatialAttention, rng::TaskLocalRNG = Random.default_rng())
    return (
        norm = initialize(m.norm, rng),
        attn = initialize(m.attn, rng) # 递归初始化 Attention 的参数
    )
end

function (m::SpatialAttention)(x::SpatialTensor{D}, ps::ParamsContainer) where D
    # x: (W, H, C, B) -> D=2
    W, H, C, B = size(x)
    
    # GroupNorm (DDPM 标配: Pre-Norm)
    h = m.norm(x, ps.norm)
    
    # 维度变换: (W, H, C, B) -> (C, W*H, B)
    # Julia 是列优先 (Column-Major) 存储：
    # 直接 reshape(x, :, B) 会合并 W, H, C，导致通道混杂。
    # 我们需要先 permutedims 把 C 放到第一维。
    # 变换路径: (W, H, C, B) -> (C, W, H, B) -> (C, W*H, B)
    h_perm = permutedims(h.data, (3, 1, 2, 4)) 
    h_flat_data = reshape(h_perm, C, W * H, B)
    
    # 包装成 SpatialTensor{1} 并调用 Attention
    # 这里的 h_seq 维度符合 Attention 的要求: (Embed, Seq, Batch)
    h_seq = SpatialTensor{1}(h_flat_data)
    
    # 调用 Attention (返回也是 SpatialTensor{1})
    attn_out_tensor = m.attn(h_seq, ps.attn)
    attn_out_data = attn_out_tensor.data # (C, W*H, B)
    
    # 维度还原: (C, W*H, B) -> (W, H, C, B)
    # 先 reshape 回 (C, W, H, B)
    out_reshaped = reshape(attn_out_data, C, W, H, B)
    # 再 permute 回 (W, H, C, B)
    out_perm = permutedims(out_reshaped, (2, 3, 1, 4))
    
    # 残差连接 (x + Attention(Norm(x)))
    # 注意: 这里的加法是 Element-wise 的，要求维度完全一致
    return x + SpatialTensor{D}(out_perm)
end

function Base.show(io::IO, m::SpatialAttention)
    print(io, "SpatialAttention(channels=$(m.attn.embed_dim))")
end
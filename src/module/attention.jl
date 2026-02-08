

"""
    Attention(embed_dim, heads; dropout=0.0, causal=false)

通用的多头注意力。
输入形状: (Embed, Seq, Batch)
"""
struct Attention{H} <: AbstractModule
    embed_dim::Int
    head_dim::Int
    scale::Float32
    # causal::Bool # 新增：控制是否应用因果遮罩
    mask::Union{Nothing,Array{Bool,3}}
    # mask 可以在 forward 时动态生成或缓存，这里为了简单先省略预计算 mask
end

function Attention{H}(embed_dim::Int, max_len::Int; causal::Bool=false) where H
    @assert embed_dim % H == 0 "Embedding dimension ($embed_dim) must be divisible by number of heads ($H)"
    head_dim = embed_dim ÷ H
    scale = Float32(1.0 / sqrt(head_dim))
    
    # 预计算 Causal Mask (Lower Triangular for Q*K' layout? No, wait below)
    # 我们将在 forward 中详细解释 Mask 的方向。
    # 简单起见，我们创建一个上三角矩阵作为"允许看见"的区域 (Permitted Area)
    # 具体的 mask 逻辑取决于 K^T * Q 还是 Q^T * K。
    # 我们采用 Julia 列优先习惯： (Key^T * Query) -> (Seq_Key, Seq_Query)
    # 我们希望 Query i 只能看见 Key j (其中 j <= i)。
    # 即 Row index j <= Col index i。这是上三角矩阵 (Upper Triangular)。
    
    if causal
        # 先创建一个 2D 矩阵 (Seq, Seq)
        full_matrix = ones(Bool, max_len, max_len)
        
        # 取上三角 (Keep Row <= Col, 即 Key <= Query)
        # triu 作用于 2D 矩阵
        causal_mask_2d = triu(full_matrix)
        
        # 变形为 3D (1, Seq, Seq) 以便在前向传播中广播
        mask = reshape(causal_mask_2d, 1, max_len, max_len)
    else
        mask = nothing
    end
    
    return return Attention{H}(
        embed_dim, 
        div(embed_dim, H), 
        Float32(1.0 / sqrt(embed_dim / H)),
        mask
    )
end
Attention(embed_dim::Int, heads::Int, max_len::Int; causal::Bool=false) = Attention{heads}(embed_dim, max_len; causal = causal)

function initialize(l::Attention{H}, rng::TaskLocalRNG = Random.default_rng()) where H
    std = sqrt(2.0f0 / (5 * l.embed_dim)) # Xavier like
    return (
        W_qkv  = randn(rng, Float32, 3 * l.embed_dim, l.embed_dim) .* Float32(std),
        W_proj = randn(rng, Float32, l.embed_dim, l.embed_dim) .* Float32(std / sqrt(2 * H)) # Residual 缩放
    )
end

function (l::Attention{H})(x::SpatialTensor{1}, ps::ParamsContainer) where H
    # x.data: (Embed, Seq, Batch)
    # T = Seq, B = Batch, D = Embed
    D, T, B = size(x.data)
    
    # QKV 投影
    # (3D, D) * (D, T*B) -> (3D, T, B)
    # 这里我们先把 x 展平 batch 维以便矩阵乘法，或者利用 Julia 的广播乘法
    # 为了最快速度，通常合并 T 和 B 做 2D 乘法，然后再 reshape
    x_flat = reshape(x.data, D, :) # (D, T*B)
    qkv = ps.W_qkv * x_flat        # (3D, T*B)
    
    qkv_reshaped = reshape(qkv, l.head_dim, H, 3, T, B)
    
    # Permute 维度: 把 3 (QKV类别) 放到最前，把 H 和 B 放到最后准备合并
    qkv_permuted = permutedims(qkv_reshaped, (3, 1, 4, 2, 5)) # (3, HeadDim, T, H, B)
    batch_dim_size = H * B
    
    q = reshape(view(qkv_permuted, 1, :, :, :, :), l.head_dim, T, batch_dim_size)
    k = reshape(view(qkv_permuted, 2, :, :, :, :), l.head_dim, T, batch_dim_size)
    v = reshape(view(qkv_permuted, 3, :, :, :, :), l.head_dim, T, batch_dim_size)
    
    kt = permutedims(k, (2, 1, 3)) 
    
    attn_scores = NNlib.batched_mul(kt, q) .* l.scale # (T, T, H*B)

    if !isnothing(l.mask)
        current_mask = view(l.mask, 1, 1:T, 1:T) # (1, T, T)
        large_neg = Float32(-1e9)
        attn_scores .+= (map(!, current_mask) .* large_neg)
    end

    attn_probs = softmax(attn_scores, dims=1)

    y = NNlib.batched_mul(v, attn_probs)
    y = reshape(permutedims(reshape(y, l.head_dim, T, H, B), (1, 3, 2, 4)), D, T * B)

    output = ps.W_proj * y # (D, T*B)

    final_out = reshape(output, D, T, B)
    return SpatialTensor{1}(final_out)
end

function (l::Attention{H})(x::SpatialTensor{2}, ps::ParamsContainer) where H
    W, H′, C, B = size(x)
    
    # 检查维度匹配
    @assert C == l.embed_dim "Input channels $C must match embed_dim $(l.embed_dim)"

    # 1. (W, H, C, B) -> (C, W*H, B)
    x_perm = permutedims(x.data, (3, 1, 2, 4))
    x_flat = reshape(x_perm, C, W*H, B)
    
    # 2. 调用核心逻辑
    out_flat = _core_attention_forward(l, SpatialTensor{1}(x_flat), ps).data
    
    # 3. 还原
    out_reshaped = reshape(out_flat, C, W, H, B)
    out_final = permutedims(out_reshaped, (2, 3, 1, 4))
    
    return SpatialTensor{2}(out_final)
end


function Base.show(io::IO, l::Attention{H}) where H
    print(io, "Attention(heads=$H, embed=$(l.embed_dim))")
end

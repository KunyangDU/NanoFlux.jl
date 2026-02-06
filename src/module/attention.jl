

"""
    Attention{H}(embed_dim::Int, seq_len::Int)

H: 头数 (Heads)，作为类型参数传入。
实现标准的 Scaled Dot-Product Attention，并强制应用 Causal Mask (GPT 风格)。

输入: (Embed, Seq, Batch)
输出: (Embed, Seq, Batch)
"""
struct Attention{H} <: AbstractModule
    embed_dim::Int
    head_dim::Int
    scale::Float32
    # 预计算的因果掩码 (1, Seq, Seq) - 使用 Bool 或 Float 都可以，这里用 Bool 方便逻辑
    # 在 NanoGPT 中通常 Seq_Len 是固定的最大上下文长度
    mask::Array{Bool, 3} 
end

function Attention{H}(embed_dim::Int, max_len::Int) where H
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
    

    # 1. 先创建一个 2D 矩阵 (Seq, Seq)
    full_matrix = ones(Bool, max_len, max_len)
    
    # 2. 取上三角 (Keep Row <= Col, 即 Key <= Query)
    # triu 作用于 2D 矩阵
    causal_mask_2d = triu(full_matrix)
    
    # 3. 变形为 3D (1, Seq, Seq) 以便在前向传播中广播
    mask = reshape(causal_mask_2d, 1, max_len, max_len)
    
    return Attention{H}(embed_dim, head_dim, scale, mask)
end
Attention(embed_dim::Int, heads::Int, max_len::Int) = Attention{heads}(embed_dim, max_len)

function initialize(l::Attention{H}, rng::TaskLocalRNG = Random.default_rng()) where H
    std = sqrt(2.0f0 / (5 * l.embed_dim)) # Xavier like
    return (
        W_qkv  = randn(rng, Float32, 3 * l.embed_dim, l.embed_dim) .* Float32(std),
        W_proj = randn(rng, Float32, l.embed_dim, l.embed_dim) .* Float32(std / sqrt(2 * H)) # Residual 缩放
    )
end


function (l::Attention{H})(x::AbstractNanoTensor, ps::ParamsContainer) where H
    # x.data: (Embed, Seq, Batch)
    # T = Seq, B = Batch, D = Embed
    D, T, B = size(x.data)
    
    # 1. QKV 投影
    # (3D, D) * (D, T*B) -> (3D, T, B)
    # 这里我们先把 x 展平 batch 维以便矩阵乘法，或者利用 Julia 的广播乘法
    # 为了最快速度，通常合并 T 和 B 做 2D 乘法，然后再 reshape
    x_flat = reshape(x.data, D, :) # (D, T*B)
    qkv = ps.W_qkv * x_flat        # (3D, T*B)
    
    # 2. 分割与重塑 (Split & Reshape heads)
    # qkv: (3 * H * HeadDim, T * B)
    # 我们需要将其变为 (HeadDim, H, 3, T*B) 以便分割
    # 但为了后续 batched_mul 方便，我们目标形状是: (HeadDim, T, H * B)
    
    # 这里的 reshape 稍微复杂，步骤如下：
    # -> (HeadDim, H, 3, T, B) 
    # -> Permute -> (3, HeadDim, T, H, B) 
    # -> Split -> Q, K, V 都是 (HeadDim, T, H*B)
    
    qkv_reshaped = reshape(qkv, l.head_dim, H, 3, T, B)
    
    # Permute 维度: 把 3 (QKV类别) 放到最前，把 H 和 B 放到最后准备合并
    qkv_permuted = permutedims(qkv_reshaped, (3, 1, 4, 2, 5)) # (3, HeadDim, T, H, B)
    
    # 此时 Q, K, V 视图分离
    # 每一个都是 (HeadDim, T, H, B)
    # 合并最后两个维度作为 "Batch" 给 batched_mul 使用 -> (HeadDim, T, H*B)
    batch_dim_size = H * B
    
    # 利用 view 避免复制
    q = reshape(view(qkv_permuted, 1, :, :, :, :), l.head_dim, T, batch_dim_size)
    k = reshape(view(qkv_permuted, 2, :, :, :, :), l.head_dim, T, batch_dim_size)
    v = reshape(view(qkv_permuted, 3, :, :, :, :), l.head_dim, T, batch_dim_size)
    
    # 3. Attention Score 计算 (Scaled Dot-Product)
    # Score = (K^T * Q) * scale
    # K: (Dh, T, BatchAll) -> K^T 实际上是指空间维度的转置
    # 我们利用 NNlib.batched_mul
    # batched_mul(A, B) 对前两维做乘法，后维是 batch
    # 我们需要 (T, T) 的结果。
    # K 的列是 key vector k_j. Q 的列是 query vector q_i.
    # Score[j, i] = k_j • q_i
    # 这等价于 K' * Q (如果把 K 看作 Matrix, 每一列是一个 Key)
    # Matrix Transpose: (T, Dh) * (Dh, T) -> (T, T)
    
    # batched_transpose 将 K 从 (Dh, T, Batch) -> (T, Dh, Batch)
    kt = permutedims(k, (2, 1, 3)) 
    
    attn_scores = NNlib.batched_mul(kt, q) .* l.scale # (T, T, H*B)
    
    # 4. Causal Masking (因果遮蔽)
    # attn_scores 形状 (Row=Key, Col=Query, Batch)
    # Query i 应该关注 Key 1...i
    # 即允许 Col i 访问 Row j (当 j <= i)
    # j <= i 是矩阵的上三角部分 (Upper Triangular)
    # 我们的 l.mask 已经是上三角为 1 (True)
    
    # 动态切片 mask 以匹配当前序列长度 T
    current_mask = view(l.mask, 1, 1:T, 1:T) # (1, T, T)
    
    # 应用 Mask: False 的地方设为 -inf
    # 广播: (T, T, H*B) + (1, T, T)
    # 注意: Julia 的 ifelse 或直接加法。为了 Zygote 友好，通常用 + (1-mask)*(-1e9)
    # 但这里用 fill value 比较直观
    # 既然 mask 是 Bool (1=Keep, 0=Mask)，我们需要把 0 变成 -inf
    large_neg = Float32(-1e9)
    masked_scores = attn_scores .+ (map(!, current_mask) .* large_neg)
    
    # 5. Softmax
    # 对 dim=1 (Key 维度, 即每一列 Query 的分布) 做 softmax
    attn_probs = softmax(masked_scores, dims=1)
    
    # 6. 加权求和
    # Out = V * Probs
    # V: (Dh, T, Batch)
    # Probs: (T, T, Batch)
    # Out: (Dh, T, Batch) -> 每一列 Out[:, i] 是 V 的列的加权和
    y = NNlib.batched_mul(v, attn_probs)
    
    # 7. 还原形状 (Merge Heads)
    # y: (HeadDim, T, H*B) -> (HeadDim, T, H, B)
    y_unflat = reshape(y, l.head_dim, T, H, B)
    
    # Permute 回去: (HeadDim, H, T, B) -> 展平前两维 -> (Embed, T, B)
    y_permuted = permutedims(y_unflat, (1, 3, 2, 4)) # (HeadDim, H, T, B)
    y_merged = reshape(y_permuted, D, T * B)
    
    # 8. 输出投影
    output = ps.W_proj * y_merged # (D, T*B)
    
    # 最终 Reshape 回 (D, T, B) 并包装
    final_out = reshape(output, D, T, B)
    return SpatialTensor{1}(final_out)
end

function Base.show(io::IO, l::Attention{H}) where H
    print(io, "Attention(heads=$H, embed=$(l.embed_dim))")
end

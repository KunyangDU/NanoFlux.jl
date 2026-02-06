"""
    Embed(vocab_size::Int, embed_dim::Int)

将整数索引序列转换为密集向量序列。
遵循 NanoFlux 的 AbstractModule 接口。

输出形状: (Embed_Dim, Seq_Len, Batch_Size) 
这是 Julia 深度学习中最优的内存布局。
"""
struct Embed <: AbstractModule
    vocab_size::Int
    embed_dim::Int
end

# 初始化权重
# 布局策略: (Embed_Dim, Vocab_Size) —— 每一列是一个词向量
function initialize(l::Embed, rng::TaskLocalRNG = Random.default_rng())
    # 缩放初始化，类似于 PyTorch 的默认行为
    scale = Float32(1.0 / sqrt(l.embed_dim))
    return (
        # 使用 Float32 保证 GPU 兼容性
        weight = randn(rng, Float32, l.embed_dim, l.vocab_size) .* scale,
    )
end

# 前向传播
# x 可以是 Vector{Int} (Batch=1) 或 Matrix{Int} (Batch > 1)
# x 的形状: (Seq_Len, Batch_Size)
function (l::Embed)(x::Union{AbstractVector{<:Integer}, AbstractMatrix{<:Integer}}, ps::ParamsContainer)
    # 利用 Julia 的列优先切片 (Slicing)
    # 这里的 ps.weight[:, x] 会生成 (Embed_Dim, Seq_Len, Batch_Size)
    # 对于 GPU (CUDA/Metal)，这会自动转换为高效的 gather 操作
    out = ps.weight[:, x]
    
    # 将其包装为 SpatialTensor{1}
    # 在 NanoFlux 语境下：
    # D=1, 数据总维度=3 -> (Channel/Embed, Spatial/Seq, Batch)
    # 这种包装让它能被 NanoFlux 的其他组件识别
    return SpatialTensor{1}(out)
end

(l::Embed)(x::AbstractArray{<:AbstractFloat}, ps::ParamsContainer) = l(floor.(Int, abs.(x)) .% l.vocab_size .+ 1, ps)
(l::Embed)(x::AbstractNanoTensor, ps::ParamsContainer) = l(x.data, ps)

# 方便打印显示
function Base.show(io::IO, l::Embed)
    print(io, "Embed($(l.vocab_size) => $(l.embed_dim))")
end

"""
    Position(embed_dim::Int, max_len::Int)

可学习的位置编码层 (Learnable Positional Embedding)。
将位置信息直接加到输入的 Embedding 上。

输入: (Embed_Dim, Seq_Len, Batch)
输出: 同输入
"""
struct Position <: AbstractModule
    embed_dim::Int
    max_len::Int
end

function initialize(l::Position, rng::TaskLocalRNG = Random.default_rng())
    # 初始化一个位置矩阵 W: (Embed, Max_Len)
    # 通常使用较小的方差初始化，以免破坏原始 Embedding 的分布
    return (
        W = randn(rng, Float32, l.embed_dim, l.max_len) .* 0.02f0,
    )
end

function (l::Position)(x::AbstractNanoTensor, ps::ParamsContainer)
    # x.data 形状预期: (Embed_Dim, Seq_Len, Batch_Size)
    # 对应 SpatialTensor{1} 的 (Channel, Spatial, Batch)
    
    seq_len = size(x, 2)
    
    if seq_len > l.max_len
        error("Sequence length ($seq_len) exceeds maximum limit ($(l.max_len)) defined in Position layer.")
    end

    # 1. 切片: 取出前 seq_len 个位置的向量
    # ps.W 形状: (Embed, Max_Len) -> 切片后: (Embed, Seq_Len)
    pos_bias = ps.W[:, 1:seq_len]

    # 2. 广播加法:
    # (Embed, Seq, Batch) + (Embed, Seq) 
    # Julia 会自动将 pos_bias 广播到每一个 Batch 上
    return SpatialTensor{1}(x.data .+ pos_bias)
end

function Base.show(io::IO, l::Position)
    print(io, "Position($(l.embed_dim), max_len=$(l.max_len))")
end

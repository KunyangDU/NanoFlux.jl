# Project Source Code Summary

## File: src/utils.jl
```julia
# relu(x::T) where T = max(zero(T), x)



```

---

## File: src/default.jl
```julia

struct CPU <: AbstractDevice end
struct GPU <: AbstractDevice end

!isdefined(@__MODULE__, :CURRENT_DEVICE) && (const CURRENT_DEVICE = Ref{AbstractDevice}(CPU()))
!isdefined(@__MODULE__, :METAL_AVAILABLE) && (const METAL_AVAILABLE = isdefined(@__MODULE__, :Metal) && Metal.functional())
!isdefined(@__MODULE__, :CUDA_AVAILABLE) && (const CUDA_AVAILABLE = isdefined(@__MODULE__, :CUDA) && CUDA.functional())


function __init__()
    # 检测 Metal 支持
    if METAL_AVAILABLE
        CURRENT_DEVICE[] = GPU() # 默认切换到 GPU
        @info "NanoFlux: Metal GPU backend detected and enabled."
    # 检测 CUDA 支持
    elseif CUDA_AVAILABLE
        CURRENT_DEVICE[] = GPU()
        @info "NanoFlux: CUDA GPU backend detected and enabled."
    end
end

__init__()

!isdefined(@__MODULE__, :TO) ? (const TO = TimerOutput()) : reset_timer!(TO)
!isdefined(@__MODULE__, :ParamsContainer) && (const ParamsContainer = Union{NamedTuple, Tuple})

manualGC() = GC.gc()

macro bg_str(s)
    # 1;32m 表示：样式1(加粗) + 颜色32(绿色)
    # 红 绿  黄
    # 31 32 33
    return "\e[1;32m" * s * "\e[0m"
end

macro g_str(s)
    return "\e[32m" * s * "\e[0m"
end
```

---

## File: src/NanoFlux.jl
```julia
using MLDatasets, MLUtils, OneHotArrays
using Zygote, NNlib
using Random
using Printf, TimerOutputs
using Statistics: mean, var
using JLD2
using LinearAlgebra: triu, dot

include("abstract.jl")

include("default.jl")

include("wrapper/tensor.jl")

include("control/algorithm.jl")
include("control/information.jl")

include("module/Identity.jl")
include("module/sequential.jl")
include("module/dense.jl")
include("module/convolution.jl")
include("module/pool.jl")
include("module/flatten.jl")
# include("module/input.jl")
include("module/attention.jl")
include("module/normalize.jl")
include("module/block.jl")
include("module/embed.jl")
include("module/utils.jl")
include("module/initialize.jl")
include("module/summary.jl")
include("module/show.jl")
include("module/resnet.jl")
include("module/time.jl")
include("module/spatial.jl")

include("network/UNetAttentionBlock.jl")
include("network/UNet.jl")

include("algorithm/train.jl")
include("algorithm/update.jl")
include("algorithm/loss.jl")
include("algorithm/generate.jl")
include("algorithm/diffusion.jl")

include("optimizer/SGD.jl")
include("optimizer/Adam.jl")

include("wrapper/interface.jl")

include("fileIO/utils.jl")
include("fileIO/tokenizer.jl")
include("fileIO/lm.jl")
include("fileIO/spatial.jl")

include("gpu/move.jl")

include("utils.jl")
```

---

## File: src/abstract.jl
```julia

abstract type AbstractAlgorithm end
abstract type AbstractInformation end
abstract type AbstractModule end
abstract type AbstractOptimizer end
abstract type AbstractState end
abstract type AbstractNanoTensor{T, N} <: AbstractArray{T, N} end
abstract type AbstractDataset end
abstract type AbstractDevice end

```

---

## File: src/gpu/move.jl
```julia
"""
    to_device(x)

将结构体 x 中的所有数组移动到 CURRENT_DEVICE[] 指定的设备上。
支持嵌套的 NamedTuple, Tuple 以及 NanoFlux 自定义的 Tensor 类型。
"""
to_device(x) = move_to(CURRENT_DEVICE[], x)

move_to(::AbstractDevice, x::Any) = x
move_to(::CPU, x::AbstractArray) = Array(x)

function move_to(::GPU, x::AbstractArray)
    is_gpu_array(x) && return x
    METAL_AVAILABLE && return Metal.mtl(convert(AbstractArray{Float32}, x))
    CUDA_AVAILABLE && return CUDA.cu(convert(AbstractArray{Float32}, x))
    error("NanoFlux Error: CURRENT_DEVICE is GPU, but no backend (Metal/CUDA) is detected!")
end

function is_gpu_array(x::AbstractArray)
    return (METAL_AVAILABLE && x isa Metal.MtlArray) || 
           (CUDA_AVAILABLE && x isa CUDA.CuArray)
end

move_to(d::AbstractDevice, x::ParamsContainer) = map(v -> move_to(d, v), x)
move_to(d::AbstractDevice, x::AbstractDict) = Dict(k => move_to(d, v) for (k,v) in x)

move_to(d::AbstractDevice, t::SpatialTensor{D}) where D = SpatialTensor{D}(move_to(d, t.data))
move_to(d::AbstractDevice, t::FlatTensor) = FlatTensor(move_to(d, t.data))

function move_to(d::AbstractDevice, st::AdamState)
    return AdamState(
        move_to(d, st.m),
        move_to(d, st.v),
        st.t
    )
end

function move_to(d::AbstractDevice, st::SGDState)
    return SGDState(
        move_to(d, st.v),
        st.t
    )
end
```

---

## File: src/optimizer/Adam.jl
```julia

@kwdef struct Adam <: AbstractOptimizer
    learning_rate::Float32    = 1e-3
    beta1::Float32 = 0.9
    beta2::Float32 = 0.999
    eps::Float32   = 1e-8
end

mutable struct AdamState{T} <: AbstractState
    m::T
    v::T
    t::Int 
end

function initialize(opt::Adam, ps::Union{NamedTuple, Tuple})
    return map(p -> initialize(opt, p), ps)
end

function initialize(::Adam, p::AbstractArray)
    return AdamState(
        zeros(eltype(p), size(p)), # m
        zeros(eltype(p), size(p)), # v
        1                          # t (初始步数)
    )
end

initialize(::Adam, ::Any) = nothing


function update!(p::AbstractArray, g::AbstractArray, state::AdamState, opt::Adam)
    m, v = state.m, state.v
    β1, β2 = opt.beta1, opt.beta2
    ϵ = opt.eps
    

    @. m = β1 * m + (1 - β1) * g
    @. v = β2 * v + (1 - β2) * (g ^ 2)
    
    correction1 = 1 - β1 ^ state.t
    correction2 = 1 - β2 ^ state.t
    step_size = opt.learning_rate * sqrt(correction2) / correction1
    
    @. p -= step_size * m / (sqrt(v) + ϵ)
end

```

---

## File: src/optimizer/SGD.jl
```julia

@kwdef struct SGD <: AbstractOptimizer
    learning_rate::Float32       = 1e-2  # 学习率
    momentum::Float32 = 0.9   # 动量系数
end
mutable struct SGDState{T<:AbstractArray} <: AbstractState
    v::T
    t::Int
end

NoOptimizer(lr::Number) = SGD(lr = convert(Float32,lr), momentum = 0.0)

function initialize(opt::SGD, ps::Union{NamedTuple, Tuple})
    return map(p -> initialize(opt, p), ps)
end

function initialize(::SGD, p::AbstractArray)
    return SGDState(
        zeros(eltype(p), size(p)), # v 初始化为全 0
        1                          # t 初始化为 1
    )
end

initialize(::SGD, ::Any) = nothing

function update!(p::AbstractArray, g::AbstractArray, state::SGDState, opt::SGD)
    @. state.v = opt.momentum * state.v + g
    @. p = p - opt.learning_rate * state.v
end
```

---

## File: src/fileIO/utils.jl
```julia

```

---

## File: src/fileIO/tokenizer.jl
```julia
"""
    SimpleTokenizer{T}

通用的简单分词器。
T: Token 的类型 (通常是 Char 或 String)。

字段:
- stoi: Token -> ID
- itos: ID -> Token
- splitter: 一个函数，定义如何将长文本切分成 Token 列表
"""
struct SimpleTokenizer{T,L}
    stoi::Dict{T, Int}
    itos::Vector{T}
    splitter::Function # 核心差异点: 切分逻辑
end

"""
    build_tokenizer(text::String; mode=:char)

构建分词器的工厂函数。
- mode=:char : 字符级 (CharTokenizer)
- mode=:word : 单词级 (WordTokenizer, 按空格切分)
"""
function build_tokenizer(text::String; mode=:char)
    if mode == :char
        splitter = s -> collect(s)
        tokens = splitter(text)
        T = Char
    elseif mode == :word
        splitter = s -> split(s) 
        tokens = String.(splitter(text)) # 转为 String
        T = String
    else
        error("Unsupported mode: $mode")
    end

    unique_tokens = sort(unique(tokens))

    stoi = Dict(t => i for (i, t) in enumerate(unique_tokens))
    itos = unique_tokens
    
    return SimpleTokenizer{T,length(itos)}(stoi, itos, splitter)
end

"""
    encode(t::SimpleTokenizer{T}, s::AbstractString)

文本 -> ID 列表
"""
function encode(t::SimpleTokenizer{T}, s::AbstractString) where T
    raw_tokens = t.splitter(s)
    
    if T == String
        raw_tokens = String.(raw_tokens)
    end
    
    return [t.stoi[token] for token in raw_tokens]
end

"""
    decode(t::SimpleTokenizer{T}, indices::AbstractVector{<:Integer})

ID 列表 -> 文本
"""
function decode(t::SimpleTokenizer{T}, indices::AbstractVector{<:Integer}) where T
    tokens = [t.itos[i] for i in indices]
    
    if T == Char
        return String(tokens)
    else
        return join(tokens, " ")
    end
end

vocab_size(::SimpleTokenizer{T,L}) where {T,L} = L

function Base.show(io::IO, t::SimpleTokenizer{T}) where T
    print(io, "SimpleTokenizer{$T}(vocab=$(vocab_size(t)))")
end
```

---

## File: src/fileIO/lm.jl
```julia

struct CausalLMDataset{T, F}
    block_size::Int
    windows::T
    transform::F # 新增：转换函数
end

function CausalLMDataset(tokens::Vector{Int}, block_size::Int; transform=identity)
    data_matrix = reshape(tokens, 1, :)
    sw = slidingwindow(data_matrix; size=block_size+1, stride=1, obsdim=2)
    return CausalLMDataset(block_size, sw, transform)
end

# --- MLUtils 接口适配 ---

MLUtils.numobs(d::CausalLMDataset) = length(d.windows)

function MLUtils.getobs(d::CausalLMDataset, i::Int)
    w = vec(d.windows[i])
    x, y = w[1:end-1], w[2:end]
    # 通过钩子允许外部注入数据增强逻辑（如 Dropout 或随机扰动）
    return d.transform((x, y))
end

function MLUtils.getobs(d::CausalLMDataset, indices::AbstractVector{<:Integer})
    samples = [MLUtils.getobs(d, i) for i in indices]
    
    batch_x = reduce(hcat, first.(samples))
    batch_y = reduce(hcat, last.(samples))
    
    return (batch_x, batch_y)
end

function Base.show(io::IO, d::CausalLMDataset)
    print(io, "CausalLMDataset(Window=$(d.block_size), Obs=$(numobs(d)))")
end
```

---

## File: src/fileIO/spatial.jl
```julia

struct SpatialDataset{D, X, Y}
    features::X
    targets::Y
    num_classes::Int
end

function SpatialDataset(x::AbstractArray, y::AbstractVector; num_classes=0, add_channel_dim=true)
    raw_dims = ndims(x)
    n_samples = size(x)[end]
    
    D = add_channel_dim ? raw_dims - 1 : raw_dims - 2
    
    x_proc = add_channel_dim ? reshape(x, size(x)[1:end-1]..., 1, n_samples) : x
    x_proc = Float32.(x_proc)
    
    y_proc = num_classes > 0 ? Float32.(onehotbatch(y, 0:(num_classes-1))) : y
    
    return SpatialDataset{D, typeof(x_proc), typeof(y_proc)}(x_proc, y_proc, num_classes)
end

MLUtils.numobs(d::SpatialDataset) = size(d.features)[end]

function MLUtils.getobs(d::SpatialDataset, i)
    idx = (ntuple(_ -> :, ndims(d.features)-1)..., i)
    return (d.features[idx...], d.targets[:, i])
end

function Base.show(io::IO, d::SpatialDataset{D}) where D
    print(io, "SpatialDataset{$(D)D}(Samples=$(numobs(d)), Classes=$(d.num_classes))")
end
```

---

## File: src/wrapper/tensor.jl
```julia

# D: 空间维度数 (例如 3D卷积 D=3)
# N: 总维度数 (自动推导为 D + 2)
struct SpatialTensor{D, T, N, A<:AbstractArray{T, N}} <: AbstractNanoTensor{T, N}
    data::A # (Spatial..., Channel, Batch)
    function SpatialTensor{D}(data::A) where {D, A}
        @assert ndims(A) == D + 2 "SpatialTensor{$D} 需要 $(D+2) 维数据: (Channel, Spatial..., Batch)"
        new{D, eltype(A), ndims(A), A}(data)
    end
end

struct FlatTensor{T, A<:AbstractArray{T, 2}} <: AbstractNanoTensor{T, 2}
    data::A # (Features, Batch)
end



```

---

## File: src/wrapper/interface.jl
```julia

Base.size(t::FlatTensor) = size(t.data)
Base.size(t::FlatTensor, d::Int) = size(t.data, d)
Base.length(t::FlatTensor) = length(t.data)
Base.getindex(t::FlatTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::FlatTensor, v, i...) = setindex!(t.data, v, i...)
Base.IndexStyle(::Type{<:FlatTensor}) = IndexLinear()
Base.eltype(::FlatTensor{T}) where T = T


function Base.show(io::IO, t::FlatTensor)
    dims = size(t)
    print(io, "FlatTensor($(eltype(t)), $(join(dims, "×")))")
end

function Base.show(io::IO, ::MIME"text/plain", t::FlatTensor)
    summary(io, t)
    println(io, ":")
    Base.print_array(io, t.data)
end

Base.summary(io::IO, t::FlatTensor) = print(io, "FlatTensor($(eltype(t)), $(join(size(t), "×")))")

function Base.similar(t::FlatTensor, ::Type{T}, dims::Dims) where {T}
    if length(dims) == 2
        return FlatTensor(similar(t.data, T, dims))
    else
        return similar(t.data, T, dims)
    end
end


Base.size(t::AbstractNanoTensor) = size(t.data)
Base.size(t::AbstractNanoTensor, d::Int) = size(t.data, d)
Base.length(t::AbstractNanoTensor) = length(t.data)

Base.getindex(t::AbstractNanoTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::AbstractNanoTensor, v, i...) = setindex!(t.data, v, i...)

Base.IndexStyle(::Type{<:AbstractNanoTensor}) = IndexLinear()

Base.eltype(::AbstractNanoTensor{T}) where T = T

function Base.show(io::IO, t::SpatialTensor{D}) where D
    dims = size(t)
    print(io, "SpatialTensor{$D}($(eltype(t)), $(join(dims, "×")))")
end

function Base.show(io::IO, t::FlatTensor)
    dims = size(t)
    print(io, "FlatTensor($(eltype(t)), $(join(dims, "×")))")
end

function Base.show(io::IO, ::MIME"text/plain", t::AbstractNanoTensor)
    print(io, typeof(t), " with size ", size(t), ":\n")
    Base.print_array(io, t.data)
end

Base.similar(t::SpatialTensor{D}, ::Type{T}, dims::Dims) where {D, T} = 
    SpatialTensor{D}(similar(t.data, T, dims))

Base.similar(t::FlatTensor, ::Type{T}, dims::Dims) where {T} = 
    FlatTensor(similar(t.data, T, dims))

Base.:+(a::SpatialTensor{D}, b::SpatialTensor{D}) where D = SpatialTensor{D}(a.data .+ b.data)
Base.:+(a::FlatTensor, b::FlatTensor) = FlatTensor(a.data .+ b.data)

```

---

## File: src/module/pool.jl
```julia


struct Pool{D, F} <: AbstractModule
    k::NTuple{D, Int}
    stride::NTuple{D, Int}
    mode::F # mean 或 maximum
end

function Pool(D::Int, k::Union{Int, NTuple}; 
              stride=k, mode=mean) # 默认 stride=k (不重叠)
    
    ks = k isa Int ? ntuple(_->k, D) : k
    st = stride isa Int ? ntuple(_->stride, D) : stride
    
    return Pool{D, typeof(mode)}(ks, st, mode)
end

for N in 1:3
    @eval begin
        # Mean Pooling
        function (l::Pool{$N})(::typeof(mean), x::SpatialTensor{$N}, ps::ParamsContainer)
            y = NNlib.meanpool(x.data, l.k; stride=l.stride)
            return SpatialTensor{$N}(y)
        end

        # Max Pooling
        function (l::Pool{$N})(::typeof(maximum), x::SpatialTensor{$N}, ps)
            y = NNlib.maxpool(x.data, l.k; stride=l.stride)
            return SpatialTensor{$N}(y)
        end
    end
end

function (l::Pool)(x::SpatialTensor, ps::ParamsContainer)
    return l(l.mode, x, ps)
end



```

---

## File: src/module/block.jl
```julia
# 建议直接叫 Block，放在 Transformer 模块下
struct Block{A, N, M} <: AbstractModule
    ln1::N    # LayerNorm
    attn::A   # Causal Attention
    ln2::N    # LayerNorm
    mlp::M    # FeedForward (MLP)
end

function Block(embed_dim::Int, heads::Int, seq_len::Int)
    # 注意：GPT 系列通常使用 GELU
    mlp_hidden = 4 * embed_dim
    return Block(
        LayerNorm(embed_dim),
        Attention(embed_dim, heads, seq_len), 
        LayerNorm(embed_dim),
        Sequential(
            # Input((embed_dim, 1)),
            Dense(embed_dim, mlp_hidden, gelu),
            Dense(mlp_hidden, embed_dim)
        )
    )
end

# GPT-2 / GPT-3 风格的 Forward (Pre-Norm)
function (m::Block)(x::AbstractNanoTensor, ps::ParamsContainer)

    x_norm1 = m.ln1(x, ps.ln1)
    attn_out = m.attn(x_norm1, ps.attn) # 这里的 attn 必须是 causal 的

    x = x + attn_out

    x_norm2 = m.ln2(x, ps.ln2)
    mlp_out = m.mlp(x_norm2, ps.mlp)

    return x + mlp_out
end

function initialize(m::Block, rng::TaskLocalRNG = Random.default_rng())
    return (
        # 这里的键名 (ln1, attn...) 必须和前向传播中 ps.xxx 的访问一致
        ln1  = initialize(m.ln1, rng),
        attn = initialize(m.attn, rng),
        ln2  = initialize(m.ln2, rng),
        mlp  = initialize(m.mlp, rng)
    )
end
```

---

## File: src/module/identity.jl
```julia
"""
    Identity(act=identity)

恒等映射模块，支持可选的激活函数。
- `Identity()`: 不做任何操作（直通），常用于 ResNet 的 Shortcut。
- `Identity(relu)`: 仅作为激活层使用。
"""
struct Identity{F} <: AbstractModule
    act::F
end

# 默认构造函数：act 默认为 identity 函数
Identity() = Identity(identity)

# 当 act 为 identity 时，直接返回输入，避免广播开销
(::Identity{typeof(identity)})(x::AbstractNanoTensor, ::ParamsContainer) = x

# 当 act 不为 identity 时 (例如 silu)，执行广播操作
function (m::Identity)(x::SpatialTensor{D}, ::ParamsContainer) where D
    # 保持 SpatialTensor 类型
    return SpatialTensor{D}(m.act.(x.data))
end

function (m::Identity)(x::FlatTensor, ::ParamsContainer)
    # 保持 FlatTensor 类型
    return FlatTensor(m.act.(x.data))
end

# 无论是否有激活函数，Identity 都没有可训练参数
initialize(::Identity, ::TaskLocalRNG) = NamedTuple()

function Base.show(io::IO, m::Identity)
    if m.act === identity
        print(io, "Identity()")
    else
        print(io, "Identity($(m.act))")
    end
end
```

---

## File: src/module/resnet.jl
```julia
# src/module/unet_blocks.jl

struct ResNetBlock <: AbstractModule
    norm1::GroupNorm
    conv1::Conv
    norm2::GroupNorm
    conv2::Conv
    time_proj::Dense # 将时间向量投影到当前通道数
    shortcut::Union{Conv, Identity} # 如果输入输出通道不同，需要 1x1 卷积
end

function ResNetBlock(in_ch::Int, out_ch::Int, time_dim::Int; groups=8)
    # Shortcut 逻辑
    shortcut = (in_ch != out_ch) ? Conv(2, in_ch, out_ch, 1) : Identity()
    
    return ResNetBlock(
        GroupNorm(groups, in_ch),
        Conv(2, in_ch, out_ch, 3, stride=1, pad=1, act=silu),
        GroupNorm(groups, out_ch),
        Conv(2, out_ch, out_ch, 3, stride=1, pad=1, act=silu),
        Dense(time_dim, out_ch, silu), # 时间投影层
        shortcut
    )
end

# 初始化逻辑略 (对每个子模块调用 initialize)

function (m::ResNetBlock)(x::SpatialTensor, t_emb::AbstractArray, ps::ParamsContainer)
    # 第一层卷积
    h = m.conv1(m.norm1(x, ps.norm1), ps.conv1)
    
    # 注入时间嵌入 (Scale & Shift 或 仅 Add)
    # DDPM 论文通常直接相加: h + dense(t_emb)
    # t_emb: (TimeDim, Batch) -> proj -> (OutCh, Batch)
    t_proj = m.time_proj(FlatTensor(t_emb), ps.time_proj).data
    
    # 广播加法: (W, H, C, B) + (1, 1, C, B)
    # 需要 reshape t_proj
    t_proj_reshaped = reshape(t_proj, 1, 1, size(t_proj)...)
    h = SpatialTensor{2}(h.data .+ t_proj_reshaped)
    
    # 第二层卷积
    h = m.conv2(m.norm2(h, ps.norm2), ps.conv2)
    
    # 残差连接
    sc = (m.shortcut isa Identity) ? x : m.shortcut(x, ps.shortcut)
    
    return h + sc
end
function initialize(m::ResNetBlock, rng::TaskLocalRNG = Random.default_rng())
    return (
        # 必须显式初始化每一个子模块，并给它们起对应的名字
        norm1     = initialize(m.norm1, rng),
        conv1     = initialize(m.conv1, rng),
        norm2     = initialize(m.norm2, rng),
        conv2     = initialize(m.conv2, rng),
        time_proj = initialize(m.time_proj, rng),
        shortcut  = initialize(m.shortcut, rng)
    )
end
```

---

## File: src/module/utils.jl
```julia

function _fmt_shape(s)
    dims = s[1:end-1] 
    return string(dims)
end

function format_number(n::Int)
    return replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
end

silu(x) = x .* sigmoid.(x) # DDPM 标配激活函数

# 最近邻上采样 (2x)
function upsample(x::SpatialTensor{D}) where D
    # 假设 x.data 是 (W, H, C, B)
    # NNlib.upsample_nearest 默认行为适配 WHCB
    new_data = NNlib.upsample_nearest(x.data, (2, 2))
    return SpatialTensor{D}(new_data)
end

# 通道拼接
function cat_channels(x1::SpatialTensor{D}, x2::SpatialTensor{D}) where D
    # 沿着第 3 维 (Channel) 拼接
    return SpatialTensor{D}(cat(x1.data, x2.data, dims=3))
end

```

---

## File: src/module/summary.jl
```julia

# 定义一个节点来存储层级信息
mutable struct InspectNode
    name::String
    type_name::String
    input_shape::String
    output_shape::String
    params_count::Int
    children::Vector{InspectNode}
    depth::Int
end

"""
    summary(model::AbstractModule, input_shape::Tuple)

对任意 NanoFlux 模型进行递归结构检查、维度推导和参数统计。
"""
function summary(model::AbstractModule, input_shape::Tuple)

    full_shape = (input_shape..., 1) 
    
    spatial_dims = length(input_shape) - 1
    if spatial_dims >= 0
        x = SpatialTensor{spatial_dims}(randn(Float32, full_shape))
    else
        x = FlatTensor(randn(Float32, full_shape))
    end

    root_node = _inspect_recursive(model, x, initialize(model, Random.default_rng()), "Model", 0)

    println("="^100)
    println("Model Inspector")
    # println("="^100)
    show(root_node)
    println("-"^100)
    total_params = _sum_params(root_node)
    println("Total Parameters: $(format_number(total_params))")
    println("="^100)
    return nothing
end

# --- 核心递归逻辑 ---

function _inspect_recursive(layer::AbstractModule, x, ps, name, depth)
    in_shape_str = _fmt_shape(size(x))

    out = try
        layer(x, ps)
    catch e
        error("Dimension Mismatch in layer [$name] ($(typeof(layer))).\nInput: $in_shape_str\nError: $e")
    end
    
    out_shape_str = _fmt_shape(size(out))

    is_container = layer isa Sequential || layer isa Block # 识别容器

    self_params = is_container ? 0 : _count_elements(ps)
    
    node = InspectNode(name, string(typeof(layer)), in_shape_str, out_shape_str, self_params, [], depth)

    if layer isa Sequential
        for (i, (sub_layer, sub_ps)) in enumerate(zip(layer.layers, ps))
            sub_node = _inspect_recursive(sub_layer, x, sub_ps, "Layer $i", depth + 1)
            push!(node.children, sub_node)
            
            x = sub_layer(x, sub_ps)
        end
    elseif layer isa Block
        # Block 结构比较特殊 (ln1, attn, ln2, mlp)
        # 这种硬编码的结构需要手动拆解
        # Block Forward: x -> ln1 -> attn -> + -> ln2 -> mlp -> +
        
        # 1. LN1
        node_ln1 = _inspect_recursive(layer.ln1, x, ps.ln1, "LN1", depth + 1)
        push!(node.children, node_ln1)
        x_norm1 = layer.ln1(x, ps.ln1)
        
        # 2. Attn
        node_attn = _inspect_recursive(layer.attn, x_norm1, ps.attn, "Attention", depth + 1)
        push!(node.children, node_attn)
        # 残差连接不改变形状，直接模拟流向
        x = x + layer.attn(x_norm1, ps.attn)
        
        # 3. LN2
        node_ln2 = _inspect_recursive(layer.ln2, x, ps.ln2, "LN2", depth + 1)
        push!(node.children, node_ln2)
        x_norm2 = layer.ln2(x, ps.ln2)
        
        # 4. MLP (通常是 Sequential)
        node_mlp = _inspect_recursive(layer.mlp, x_norm2, ps.mlp, "MLP", depth + 1)
        push!(node.children, node_mlp)
    # else
        # @warn "module summary not defined!"
    end
    
    return node
end

function _sum_params(node::InspectNode)
    c = node.params_count
    for child in node.children
        c += _sum_params(child)
    end
    return c
end

_count_elements(x::Union{NamedTuple, Tuple}) = sum(_count_elements, x; init=0)
_count_elements(x::AbstractArray) = length(x)
_count_elements(x::Any) = 0
```

---

## File: src/module/sequential.jl
```julia
struct Sequential{T} <: AbstractModule
    layers::T
    function Sequential(layers...)
        S = new{typeof(layers)}(layers)
        # _check(S)
        return S
    end
end

Base.iterate(S::Sequential) = iterate(S.layers)
Base.iterate(S::Sequential, state) = iterate(S.layers, state)
Base.length(S::Sequential) = length(S.layers)
Base.getindex(S::Sequential, i) = getindex(S.layers, i)
Base.lastindex(S::Sequential) = lastindex(S.layers)
Base.eltype(::Type{Sequential}) = AbstractModule


# function (model::Sequential)(x, ps::ParamsContainer)
#     for (i, layer) in enumerate(model.layers)
#         x = layer(x, ps[i])
#     end
#     return x
# end

@inline function chain(x, layers::Tuple, ps::Union{Tuple, NamedTuple})
    layer = layers[1]
    p = ps[1]
    out = layer(x, p)
    return chain(out, Base.tail(layers), Base.tail(ps))
end

@inline chain(x, ::Tuple{}, ::Union{Tuple, NamedTuple}) = x

(model::Sequential)(x, ps::Union{Tuple, NamedTuple}) = chain(x, model.layers, ps)



function initialize(model::Sequential, rng::TaskLocalRNG = Random.default_rng())
    params_tuple = ntuple(i -> initialize(model.layers[i], rng), length(model.layers))
    return params_tuple
end
```

---

## File: src/module/normalize.jl
```julia

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
```

---

## File: src/module/dense.jl
```julia

struct Dense{F} <: AbstractModule
    in_dim::Int
    out_dim::Int
    act::F
end
Dense(in::Int, out::Int, act=identity) = Dense{typeof(act)}(in, out, act)

(l::Dense)(x::FlatTensor, ps::ParamsContainer) = FlatTensor(l.act.(ps.W * x.data .+ ps.b))

function (l::Dense)(x::SpatialTensor{D}, ps::ParamsContainer) where D
    # x.data 形状: (Features, Spatial..., Batch)
    # 对于 GPT: (Embed, Seq, Batch) -> D=1
    
    raw_data = x.data
    sz = size(raw_data)
    in_features = sz[1]
    
    # 检查维度匹配
    if in_features != l.in_dim
        error("Dense Layer Mismatch! Expected input dim $(l.in_dim), got $in_features")
    end

    # 策略：这是 Pointwise 操作 (对每个点独立做 Dense)
    # 我们把 (Features, A, B, C...) 视为 (Features, Total_Points)
    # 这样可以用一次矩阵乘法完成所有计算，效率最高
    
    flat_input = reshape(raw_data, in_features, :) # (In, N)
    
    # (Out, In) * (In, N) -> (Out, N)
    flat_out = ps.W * flat_input .+ ps.b
    flat_act = l.act.(flat_out)
    
    # 把第一维从 In 变成 Out，后面的维度保持原样
    new_size = (l.out_dim, sz[2:end]...)
    final_data = reshape(flat_act, new_size)
    
    return SpatialTensor{D}(final_data)
end

function Base.show(io::IO, l::Dense)
    print(io, "Dense($(l.in_dim) => $(l.out_dim))")
    l.act != identity && print(io, ", $(l.act)")
end


function initialize(l::Dense, rng::TaskLocalRNG = Random.default_rng())
    scale = sqrt(2.0f0 / l.in_dim)
    return (
        W = randn(rng, Float32, l.out_dim, l.in_dim) .* scale,
        b = zeros(Float32, l.out_dim)
    )
end

```

---

## File: src/module/spatial.jl
```julia
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
```

---

## File: src/module/show.jl
```julia


function Base.show(root::InspectNode)
    # 1. 打印表头 (Header)
    println("-"^100)
    @printf("%-45s %-22s %-22s %-10s\n", "Layer (Type)", "Input", "Output", "Param #")
    println("-"^100)

    _print_tree_recursive(root, "", true)

end


function _print_tree_recursive(node::InspectNode, prefix::String, is_last::Bool)

    if node.depth == 0
        connector = ""
        current_prefix = ""
    else
        connector = is_last ? "└─ " : "├─ "
        current_prefix = prefix
    end

    # 简化类型名：只取 Struct 名，去掉 {T...}
    simple_type = split(node.type_name, "{")[1]
    
    display_name = isempty(node.name) ? simple_type : "$(node.name) ($(simple_type))"
    
    tree_str = current_prefix * connector * display_name
    
    if length(tree_str) > 38
        tree_str = tree_str[1:35] * "..."
    end

    param_str = node.params_count > 0 ? format_number(node.params_count) : ""

    @printf("%-45s %-22s %-22s %-10s\n", 
            tree_str, 
            node.input_shape, 
            node.output_shape, 
            param_str)

    children = node.children
    count = length(children)
    
    if node.depth == 0
        next_prefix = "" # 根节点之下直接开始
    else
        next_prefix = prefix * (is_last ? "   " : "│  ")
    end

    for (i, child) in enumerate(children)
        is_last_child = (i == count)
        
        # 递归调用
        _print_tree_recursive(child, next_prefix, is_last_child)
    end
end
```

---

## File: src/module/attention.jl
```julia


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

```

---

## File: src/module/time.jl
```julia
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
```

---

## File: src/module/initialize.jl
```julia

initialize(::AbstractModule,rng::TaskLocalRNG) = NamedTuple()


```

---

## File: src/module/embed.jl
```julia
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

    pos_bias = ps.W[:, 1:seq_len]

    return SpatialTensor{1}(x.data .+ pos_bias)
end

function Base.show(io::IO, l::Position)
    print(io, "Position($(l.embed_dim), max_len=$(l.max_len))")
end

```

---

## File: src/module/flatten.jl
```julia

"""
    Flatten()

将任意维度的 SpatialTensor 展平为 FlatTensor。
输入: (Channel, D1, D2..., Batch)
输出: (Features, Batch)
"""
struct Flatten <: AbstractModule end


function (::Flatten)(x::SpatialTensor{D}, ::ParamsContainer) where D
    batch_size = size(x.data)[end]
    flat_data = reshape(x.data, :, batch_size)
    return FlatTensor(flat_data)
end

(::Flatten)(x::FlatTensor, ::ParamsContainer) = x

```

---

## File: src/module/convolution.jl
```julia

struct Conv{D, F, P} <: AbstractModule
    in_ch::Int   # 需要记录这些以进行初始化
    out_ch::Int
    k_size::NTuple{D, Int}
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    pad::P  # 新增: Padding 参数
    act::F
end

function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; 
              stride=1, dilation=1, pad=0, act=identity)

    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    pd = pad isa Int ? ntuple(_->pad, D) : pad
    
    return Conv{D, typeof(act), typeof(pd)}(in_ch, out_ch, ks, st, di, pd, act)
end
for N in 1:3    
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N}, ps::ParamsContainer)
            y = NNlib.conv(x.data, ps.W; stride=l.stride, dilation=l.dilation, pad=l.pad)
            bias_shape = (ntuple(_->1, $N)..., length(ps.b), 1)
            return SpatialTensor{$N}(l.act.(y .+ reshape(ps.b, bias_shape)))
        end
    end
end

function initialize(l::Conv{D}, rng::TaskLocalRNG = Random.default_rng()) where {D}
    w_shape = (l.k_size..., l.in_ch, l.out_ch) 
    fan_in = l.in_ch * prod(l.k_size)
    scale = sqrt(2.0 / fan_in)
    
    return (
        W = randn(rng, Float32, w_shape...) .* Float32(scale),
        b = zeros(Float32, l.out_ch)
    )
end

function Base.show(io::IO, l::Conv)
    print(io, "Conv($(l.in_ch) => $(l.out_ch), k=$(l.k_size)")
    all(x->x==1, l.stride) || print(io, ", s=$(l.stride)")
    all(x->x==0, l.pad)    || print(io, ", p=$(l.pad)") # 打印 padding
    all(x->x==1, l.dilation) || print(io, ", d=$(l.dilation)")
    l.act != identity && print(io, ", $(l.act)")
    print(io, ")")
end


```

---

## File: src/network/UNetAttentionBlock.jl
```julia
struct UNetAttentionBlock <: AbstractModule
    norm::GroupNorm
    attn::Attention # 只要这一个 struct
end

function UNetAttentionBlock(channels::Int)
    return UNetAttentionBlock(
        GroupNorm(32, channels),
        Attention(channels, 4; causal=false) # 自动适配图像
    )
end

function (m::UNetAttentionBlock)(x::SpatialTensor, ps::ParamsContainer)
    # x 是 (W, H, C, B)
    # m.norm 处理 (W, H, C, B)
    h = m.norm(x, ps.norm)
    
    # m.attn 自动识别这是 SpatialTensor{2}，自动 flatten 处理
    h = m.attn(h, ps.attn)
    
    # 残差连接
    return x + h
end
```

---

## File: src/network/UNet.jl
```julia
struct UNet <: AbstractModule
    time_embed::TimeEmbedding
    
    # 简单的两层架构示例 (32 -> 16 -> 8)
    inc::Conv
    
    # Down 1 (32 -> 16)
    down1_1::ResNetBlock
    down1_2::ResNetBlock
    down1_sample::Conv # stride=2
    
    # Down 2 (16 -> 8)
    down2_1::ResNetBlock
    down2_2::ResNetBlock
    down2_attn::SpatialAttention # 在 8x8 处加 Attention
    down2_sample::Conv # stride=2
    
    # Middle (8x8)
    mid_1::ResNetBlock
    mid_attn::SpatialAttention
    mid_2::ResNetBlock
    
    # Up 2 (8 -> 16)
    up2_0::Conv # Upsample
    up2_1::ResNetBlock # Input: 拼接后的通道
    up2_2::ResNetBlock
    
    # Up 1 (16 -> 32)
    up1_0::Conv
    up1_1::ResNetBlock
    up1_2::ResNetBlock
    
    out::Sequential # Norm -> SiLU -> Conv
end

# Forward 逻辑需要手动管理 Skip Connections
function (m::UNet)(x::SpatialTensor, t::Vector{Int}, ps::ParamsContainer)
    # 1. Time Embed
    t_emb = m.time_embed(t, ps.time_embed)
    
    # 2. Input
    h_inc = m.inc(x, ps.inc)
    
    # 3. Down 1 (32x32)
    h_d1 = m.down1_1(h_inc, t_emb, ps.down1_1)
    h_d1 = m.down1_2(h_d1, t_emb, ps.down1_2)
    # [关键] 保存 h_d1 用于后面的 skip connection，而不是 push!
    
    h_d1_sample = m.down1_sample(h_d1, ps.down1_sample) # 下采样 -> 16x16
    
    # 4. Down 2 (16x16)
    h_d2 = m.down2_1(h_d1_sample, t_emb, ps.down2_1)
    h_d2 = m.down2_2(h_d2, t_emb, ps.down2_2)
    h_d2 = m.down2_attn(h_d2, ps.down2_attn)
    # [关键] 保存 h_d2 用于后面的 skip connection
    
    h_d2_sample = m.down2_sample(h_d2, ps.down2_sample) # 下采样 -> 8x8
    
    # 5. Middle (8x8)
    h_mid = m.mid_1(h_d2_sample, t_emb, ps.mid_1)
    h_mid = m.mid_attn(h_mid, ps.mid_attn)
    h_mid = m.mid_2(h_mid, t_emb, ps.mid_2)
    
    # 6. Up 2 (8 -> 16)
    h_up2 = upsample(h_mid)
    h_up2 = m.up2_0(h_up2, ps.up2_0)
    
    # [关键] 直接使用 h_d2 进行拼接，而不是 pop!
    h_up2 = cat_channels(h_up2, h_d2)
    
    h_up2 = m.up2_1(h_up2, t_emb, ps.up2_1)
    h_up2 = m.up2_2(h_up2, t_emb, ps.up2_2)
    
    # 7. Up 1 (16 -> 32)
    h_up1 = upsample(h_up2)
    h_up1 = m.up1_0(h_up1, ps.up1_0)
    
    # [关键] 直接使用 h_d1 进行拼接
    h_up1 = cat_channels(h_up1, h_d1)
    
    h_up1 = m.up1_1(h_up1, t_emb, ps.up1_1)
    h_up1 = m.up1_2(h_up1, t_emb, ps.up1_2)
    
    # 8. Out
    return m.out(h_up1, ps.out)
end

# 放在 src/network/UNet.jl 中

function initialize(m::UNet, rng::TaskLocalRNG = Random.default_rng())
    return (
        time_embed   = initialize(m.time_embed, rng),
        inc          = initialize(m.inc, rng),
        
        # Down 1
        down1_1      = initialize(m.down1_1, rng),
        down1_2      = initialize(m.down1_2, rng),
        down1_sample = initialize(m.down1_sample, rng),
        
        # Down 2
        down2_1      = initialize(m.down2_1, rng),
        down2_2      = initialize(m.down2_2, rng),
        down2_attn   = initialize(m.down2_attn, rng),
        down2_sample = initialize(m.down2_sample, rng),
        
        # Middle
        mid_1        = initialize(m.mid_1, rng),
        mid_attn     = initialize(m.mid_attn, rng),
        mid_2        = initialize(m.mid_2, rng),
        
        # Up 2
        up2_0        = initialize(m.up2_0, rng),
        up2_1        = initialize(m.up2_1, rng),
        up2_2        = initialize(m.up2_2, rng),
        
        # Up 1
        up1_0        = initialize(m.up1_0, rng),
        up1_1        = initialize(m.up1_1, rng),
        up1_2        = initialize(m.up1_2, rng),
        
        # Out
        out          = initialize(m.out, rng)
    )
end
```

---

## File: src/algorithm/loss.jl
```julia
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
```

---

## File: src/algorithm/train.jl
```julia

function train!(model::AbstractModule, ps::ParamsContainer,
                    train_loader::DataLoader, 
                    opt::AbstractOptimizer, 
                    config::TrainerConfig)
    history = TrainingHistory()
    opt_state = to_device(initialize(opt, ps))
    manualGC()

    total_loaders = length(train_loader)

    for epoch in 1:config.epochs
        history.count > config.cut_step && break
        for (x_raw, y_raw) in train_loader
            history.count > config.cut_step && break
            x_raw = to_device(x_raw)
            y_raw = to_device(y_raw)
            
            @timeit TO "Data Prepare" begin
                ndims_spatial = ndims(x_raw) - 2
                x = SpatialTensor{ndims_spatial}(x_raw)
                y = y_raw
            end

            @timeit TO "Back Propagation" _train_step!(model, x, y, ps, opt_state, opt, history, config.config)

            if config.show_times > 0 && mod(mod(history.count - 1, total_loaders) + 1, config.show_times) == 0
                print("Epoch $(epoch) [$(mod(history.count-1, total_loaders) + 1)/$(total_loaders)] - ")
                show(history)
            end

            history.count += 1
        end 
        history.avg_loss = mean(history.loss[max(1, length(history.loss) - length(train_loader) + 1):end])
        history.avg_acc = mean(history.accuracy[max(1, length(history.accuracy) - length(train_loader) + 1):end])

        @timeit TO "gc" manualGC()

        show(TO;title = "$(epoch) / $(config.epochs)")
        print("\n")
        show(history)

        if config.target_loss !== nothing && history.avg_loss  <= config.target_loss
            if history.count_loss ≥ config.patience
                println()
                println(bg"Target Loss Reached!"," ($(history.avg_loss) <= $(config.target_loss))")
                println("Stopping training early at Epoch $epoch.")
                break
            else
                history.count_loss += 1
            end
        end

        if config.target_acc !== nothing && history.avg_acc >= config.target_acc
            if history.count_acc ≥ config.patience
                println()
                println(bg"Target Accuracy Reached!"," ($(history.avg_acc) >= $(config.target_acc))")
                println("Stopping training early at Epoch $epoch.")
                break
            else
                history.count_acc += 1
            end
        end
    end
    
    show(TO)
    print("\n")
    
    return ps,history
end


function _train_step!(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, 
                        ps::ParamsContainer,
                        opt_state::ParamsContainer,
                        opt::AbstractOptimizer,
                        history::TrainingHistory,
                        config::AbstractAlgorithm)

    @timeit TO "calc gradient" loss_val, grads = Zygote.withgradient(p -> loss(model, x, y, p, config), ps)

    @timeit TO "update!" update!(ps, grads[1], opt_state, opt)

    if isdefined(history, :loss)
        push!(history.loss, loss_val)
        push!(history.accuracy, accuracy(model, x, y, ps, config))
    end

end



```

---

## File: src/algorithm/diffusion.jl
```julia
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
```

---

## File: src/algorithm/update.jl
```julia

function update!(ps::ParamsContainer, gs::ParamsContainer, states::ParamsContainer, opt::AbstractOptimizer)
    map(ps, gs, states) do p, g, s
        update!(p, g, s, opt)
        if s isa AbstractState 
            s.t += 1
            # opt.learning_rate * (s.t - 1)/s.t
        end
    end
    return nothing
end

update!(::NamedTuple{(), Tuple{}}, ::Any, ::Any, opt::AbstractOptimizer) = nothing
```

---

## File: src/algorithm/generate.jl
```julia
# src/algorithm/generate.jl

"""
    generate(model, ps, tokenizer, prompt; max_new_tokens=50, temperature=1.0)

自回归文本生成。
"""
function generate(model, ps, tokenizer, prompt::String; 
                  max_new_tokens::Int=50, temperature::Float32=1.0f0, block_size::Int=32)
    
    # 1. 编码 Prompt
    # ids: Vector{Int}
    input_ids = encode(tokenizer, prompt)
    
    # 确保不为空
    if isempty(input_ids)
        input_ids = [1] # fallback
    end

    for _ in 1:max_new_tokens
        # 2. 截断上下文 (不能超过 block_size)
        # 如果当前序列太长，只取最后 block_size 个 token
        cond_idx = max(1, length(input_ids) - block_size + 1)
        idx_cond = input_ids[cond_idx:end]
        
        # 3. 准备输入 (Seq, Batch=1)
        # reshape 为 (Seq, 1) 以适配 src/algorithm/train.jl 中的维度逻辑
        x = reshape(idx_cond, :, 1)
        
        # 4. 前向传播
        # model 返回 SpatialTensor{1}, .data 为 (Vocab, Seq, 1)
        # 我们这里手动包装一下或者让 Embed 处理，根据 src/module/embed.jl，Embed 可以直接处理 Array
        # 但为了通过 Sequential 的 _check 或保持一致性，我们传入 Matrix{Int}
        
        logits = model(x, ps).data
        
        # 5. 取最后一个时间步的预测 (Predict Next Token)
        # (Vocab, 1)
        next_token_logits = logits[:, end, 1] ./ temperature
        
        # 6. 采样 (Sampling)
        # 为了简单，这里用贪婪采样 (Greedy): 直接取最大值
        next_token = argmax(next_token_logits)
        
        # 7. 拼接
        push!(input_ids, next_token)
    end
    
    # 8. 解码回文本
    return decode(tokenizer, input_ids)
end
```

---

## File: src/control/algorithm.jl
```julia
struct NoAlgorithm <: AbstractAlgorithm end
"""
    SimpleAlgorithm(args...)

配置训练的超参数。
支持关键字构建，例如: `SimpleAlgorithm(learning_rate=1e-3, epochs=20)`
"""
@kwdef struct TrainerConfig <: AbstractAlgorithm
    epochs::Int                          = 10
    batch_size::Int                      = 32
    show_times::Int                      = 1
    target_loss::Union{Float32, Nothing} = nothing 
    target_acc::Union{Float32, Nothing}  = nothing
    patience::Int64                      = 1
    cut_step::Number                     = Inf
    config::AbstractAlgorithm            = NoAlgorithm()
end



struct DiffusionProcess <: AbstractAlgorithm
    timesteps::Int
    beta::Vector{Float32}
    alpha::Vector{Float32}
    alpha_bar::Vector{Float32}
    sqrt_alpha_bar::Vector{Float32}
    sqrt_one_minus_alpha_bar::Vector{Float32}
end

function DiffusionProcess(timesteps::Int=1000; beta_start::Float64=1e-4, beta_end::Float64=0.02)
    # 线性调度
    beta = range(beta_start, beta_end, length=timesteps) |> collect .|> Float32
    alpha = 1.0f0 .- beta
    alpha_bar = cumprod(alpha)
    
    return DiffusionProcess(
        timesteps,
        beta,
        alpha,
        alpha_bar,
        sqrt.(alpha_bar),
        sqrt.(1.0f0 .- alpha_bar)
    )
end
```

---

## File: src/control/information.jl
```julia


@kwdef mutable struct TrainingHistory <: AbstractInformation
    loss::Vector{Float64}        = Float64[]
    accuracy::Vector{Float64}    = Float64[]
    count::Int64         = 1
    count_loss::Int64    = 1
    count_acc::Int64     = 1
    avg_loss::Float64    = Inf
    avg_acc::Float64     = 0.0
    to::TimerOutput      = TO
end

function Base.show(io::IO, h::TrainingHistory)
    if isempty(h.loss)
        print(io, "TrainingHistory (empty)")
    else
        @printf(io, "Loss: %10.6f  Accuracy: %10.6f  AvgLoss: %10.6f  AvgAcc: %10.6f (Steps: %d)", 
                h.loss[end], h.accuracy[end], h.avg_loss, h.avg_acc, length(h.loss))
    end
    print("\n")
end


```

---


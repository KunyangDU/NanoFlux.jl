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
using Statistics: mean
using JLD2
using LinearAlgebra: triu, dot

include("abstract.jl")

include("default.jl")

include("wrapper/tensor.jl")

include("control/algorithm.jl")
include("control/information.jl")

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

include("algorithm/train.jl")
include("algorithm/update.jl")
include("algorithm/loss.jl")
include("algorithm/generate.jl")

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
    # 1. 路径 A: Norm -> Attention
    # 注意：x 保持原样进入残差，而 norm 后的数据进入计算
    x_norm1 = m.ln1(x, ps.ln1)
    attn_out = m.attn(x_norm1, ps.attn) # 这里的 attn 必须是 causal 的
    
    # 2. 残差连接 1
    x = x + attn_out
    
    # 3. 路径 B: Norm -> MLP
    x_norm2 = m.ln2(x, ps.ln2)
    mlp_out = m.mlp(x_norm2, ps.mlp)
    
    # 4. 残差连接 2
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

## File: src/module/utils.jl
```julia

function _fmt_shape(s)
    dims = s[1:end-1] 
    return string(dims)
end

function format_number(n::Int)
    return replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
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
    
    # 1. 融合后端维度
    flat_input = reshape(raw_data, in_features, :) # (In, N)
    
    # 2. 矩阵乘法
    # (Out, In) * (In, N) -> (Out, N)
    flat_out = ps.W * flat_input .+ ps.b
    
    # 3. 激活函数
    flat_act = l.act.(flat_out)
    
    # 4. 还原形状
    # 把第一维从 In 变成 Out，后面的维度保持原样
    new_size = (l.out_dim, sz[2:end]...)
    final_data = reshape(flat_act, new_size)
    
    return SpatialTensor{D}(final_data)
end

function Base.show(io::IO, l::Dense)
    print(io, "Dense($(l.in_dim) => $(l.out_dim))")
    l.act != identity && print(io, ", $(l.act)")
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

```

---

## File: src/module/initialize.jl
```julia

function initialize(model::Sequential, rng::TaskLocalRNG = Random.default_rng())
    params_tuple = ntuple(i -> initialize(model.layers[i], rng), length(model.layers))
    return params_tuple
end

function initialize(l::Conv{D}, rng::TaskLocalRNG = Random.default_rng()) where D
    w_shape = (l.k_size..., l.in_ch, l.out_ch) 
    fan_in = l.in_ch * prod(l.k_size)
    scale = sqrt(2.0 / fan_in)
    
    return (
        W = randn(rng, Float32, w_shape...) .* Float32(scale),
        b = zeros(Float32, l.out_ch)
    )
end

function initialize(l::Dense, rng::TaskLocalRNG = Random.default_rng())
    scale = sqrt(2.0f0 / l.in_dim)
    return (
        W = randn(rng, Float32, l.out_dim, l.in_dim) .* scale,
        b = zeros(Float32, l.out_dim)
    )
end

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

struct Conv{D, F} <: AbstractModule
    in_ch::Int   # 需要记录这些以进行初始化
    out_ch::Int
    k_size::NTuple{D, Int}
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    act::F
end

function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; stride=1, dilation=1, act=identity)
    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    return Conv{D, typeof(act)}(in_ch, out_ch, ks, st, di, act)
end

for N in 1:3    
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N}, ps::ParamsContainer)
            y = NNlib.conv(x.data, ps.W; stride=l.stride, dilation=l.dilation, pad=0)
            bias_shape = (ntuple(_->1, $N)..., length(ps.b), 1)
            return SpatialTensor{$N}(l.act.(y .+ reshape(ps.b, bias_shape)))
        end
    end
end


```

---

## File: src/algorithm/loss.jl
```julia
function loss(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer)
    y_pred = model(x, ps)
    logits = y_pred.data
    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    return -sum(y .* log.(probs .+ 1.0f-10)) / size(logits, 2)
end

function loss(model::AbstractModule, x::AbstractNanoTensor, y::Matrix{Int}, ps::ParamsContainer)
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
function accuracy(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer)
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
function accuracy(model::AbstractModule, x::AbstractNanoTensor, y::Matrix{Int}, ps::ParamsContainer)
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

            @timeit TO "Back Propagation" _train_step!(model, x, y, ps, opt_state, opt, history)

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
                        history::TrainingHistory)

    @timeit TO "calc gradient" loss_val, grads = Zygote.withgradient(p -> loss(model, x, y, p), ps)

    @timeit TO "update!" update!(ps, grads[1], opt_state, opt)

    if isdefined(history, :loss)
        push!(history.loss, loss_val)
        push!(history.accuracy, accuracy(model, x, y, ps))
    end

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


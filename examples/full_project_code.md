# Project Source Code Summary

## File: src/utils.jl
```julia
relu(x::T) where T = max(zero(T), x)



```

---

## File: src/default.jl
```julia

if !isdefined(@__MODULE__, :TO)
    const TO = TimerOutput()
else
    reset_timer!(TO)
end

if !isdefined(@__MODULE__, :ParamsContainer)
    const ParamsContainer = Union{NamedTuple, Tuple}
end

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
using Metal
using Statistics: mean

include("default.jl")

include("abstract.jl")

include("wrapper/tensor.jl")

include("control/algorithm.jl")
include("control/information.jl")

include("module/sequential.jl")
include("module/dense.jl")
include("module/convolution.jl")
include("module/pool.jl")
include("module/flatten.jl")
include("module/input.jl")
include("module/check.jl")
include("module/utils.jl")
include("module/initialize.jl")

include("fileIO/utils.jl")
include("fileIO/mnist.jl")

include("algorithm/train.jl")
include("algorithm/update.jl")
include("algorithm/loss.jl")

include("optimizer/SGD.jl")
include("optimizer/Adam.jl")

include("wrapper/interface.jl")

include("fileIO/utils.jl")
include("fileIO/mnist.jl")
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

## File: src/fileIO/mnist.jl
```julia
"""
数据加载器构建函数
负责下载、维度重塑 (增加通道)、类型转换和分批
"""
function mnist_loader(batch_size::Int)
    train_x_raw, train_y_raw = MNIST(split=:train)[:]
    x_reshaped = reshape(train_x_raw, 28, 28, 1, :)
    x_data = Float32.(x_reshaped)
    y_oh = onehotbatch(train_y_raw, 0:9)
    y_data = Float32.(y_oh)
    loader = DataLoader((x_data, y_data), batchsize=batch_size, shuffle=true)
    return loader
end
```

---

## File: src/fileIO/utils.jl
```julia

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

## File: src/module/sequential.jl
```julia
struct Sequential{T} <: AbstractModule
    layers::T
    function Sequential(layers...)
        S = new{typeof(layers)}(layers)
        _check(S)
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

## File: src/module/input.jl
```julia
"""
    Input(shape::Tuple)

一个虚拟层，仅用于在 model_summary 中记录输入形状。
在前向传播中，它是什么都不做的直通车 (Identity)。
"""
struct Input <: AbstractModule
    shape::Tuple
end
Input(ds...) = Input(Tuple(ds))
(l::Input)(x, ps::ParamsContainer) = x

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

function (l::Dense)(x::SpatialTensor, ps::ParamsContainer)
    raw_data = x.data
    full_size = size(raw_data)
    batch_size = full_size[end]
    
    flat_features = length(raw_data) ÷ batch_size
    
    expected_in = size(ps.W, 2)
    
    if flat_features != expected_in
        error(b"❌ Auto-Flatten Dimension Mismatch!\n" *
              "Dense layer expects input dim: $(expected_in)\n" *
              "But incoming tensor $(full_size) flattens to: $(flat_features)\n" *
              "Check your Conv/Pool parameters.")
    end

    flat_data = reshape(raw_data, flat_features, batch_size)

    return l(FlatTensor(flat_data), ps::ParamsContainer)
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

## File: src/module/check.jl
```julia


"""
    _check(model, input_shape)

显式梯度架构下的模型检查器。
它会通过生成临时参数并运行一次前向传播来验证形状匹配。
"""
function _check(model::Sequential, input_shape::Union{Tuple, Nothing}=nothing)
    println("="^80)
    println("Model Architecture Inspector")
    println("="^80)

    layers = model.layers
    if input_shape === nothing
        if layers[1] isa Input
            input_shape = layers[1].shape
        else
            error("Missing input_shape! \nPlease provide it as an argument OR add an Input(shape) layer at the start of your model.")
        end
    end

    spatial_dims = length(input_shape) - 1
    if spatial_dims < 1
        error("Input shape must be at least (Channel, Len...), got $input_shape")
    end

    full_shape = (input_shape..., 1) # Batch=1
    x_data = randn(Float32, full_shape)

    x = SpatialTensor{spatial_dims}(x_data)
    
    rng = Random.default_rng()
    full_ps = initialize(model, rng) 

    println(@sprintf("Input Signal: %s (Batch=1)", string(size(x))))
    println("-"^80)
    @printf("%-4s %-15s %-25s %-25s %-10s\n", "ID", "Layer Type", "Input Shape", "Output Shape", "Params")
    println("-"^80)

    total_params = 0
    
    for (i, (layer, layer_ps)) in enumerate(zip(layers, full_ps))
        layer_type = string(typeof(layer))
        layer_name = split(layer_type, "{")[1]
        in_shape = size(x)
        
        try
            out = layer(x, layer_ps)
            
            out_shape = size(out)
            
            n_params = _count_elements(layer_ps)
            total_params += n_params
            
            str_in  = _fmt_shape(in_shape)
            str_out = _fmt_shape(out_shape)
            
            @printf("%-4d %-15s %-25s %-25s %-10s\n", 
                    i, layer_name, str_in, str_out, format_number(n_params))
            
            x = out
            
        catch e
            println("\n" * "!"^80)
            println("Layer Dimension Mismatch Detected at Layer $i [$layer_name]!")
            println("!"^80)
            println("   Expected Input: Compatible with $(_fmt_shape(in_shape))")
            
            
            if layer isa Dense
                if haskey(layer_ps, :W)
                    expected_dim = size(layer_ps.W, 2)
                    println("   Layer Config:   InputDim = $expected_dim")
                    println("   Analysis:       The Dense layer expects $expected_dim features, but received $(in_shape[1]).")
                    println("                   (Did you calculate the Flatten output size correctly?)")
                end
            elseif layer isa Conv
                println("   Analysis:       Convolution failure.")
                println("                   Check if input spatial size is smaller than Kernel size.")
            end
            
            println("\nERROR DETAIL:")
            showerror(stdout, e)
            println()
            return
        end
    end
    
    println("-"^80)
    println(g"CHECK PASSED")
    println("Total Parameters: $(format_number(total_params))")
    println("="^80)
end

_check(layers::Vector{<:AbstractModule}, input_shape::Union{Tuple, Nothing}=nothing) = _check(Sequential(layers...), input_shape)

_count_elements(x::Union{NamedTuple, Tuple}) = sum(_count_elements, x; init=0)
_count_elements(x::AbstractArray) = length(x)
_count_elements(x::Any) = 0
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
            
            # 处理 Bias
            # y 的形状是 (W_out, H_out, C_out, B)
            # ps.b 的形状是 (C_out,)
            # 我们需要将 b reshape 为 (1, 1, C_out, 1) 才能正确广播
            
            # 生成 reshape 维度: 前面 N 个 1，中间是 C，最后是 1
            # 例如 2D 卷积: (1, 1, C_out, 1)
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
    return -sum(y .* log.(probs .+ 1e-10)) / size(logits, 2)
end

function accuracy(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer)
    y_pred = model(x, ps)
    logits = y_pred.data
    pred_idx = [c[1] for c in argmax(logits, dims=1)]
    true_idx = [c[1] for c in argmax(y, dims=1)]
    return mean(pred_idx .== true_idx)
end
```

---

## File: src/algorithm/train.jl
```julia

function train!(model::AbstractModule, train_loader::DataLoader, opt::AbstractOptimizer, config::TrainerConfig)

    ps = initialize(model)
    history = TrainingHistory()
    opt_state = initialize(opt, ps)
    manualGC()

    total_loaders = length(train_loader)

    for epoch in 1:config.epochs
        for (x_raw, y_raw) in train_loader

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
        history.avg_loss = mean(history.loss[end - length(train_loader) + 1:end])
        history.avg_acc = mean(history.accuracy[end - length(train_loader) + 1:end])

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
    
    return history
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
        s isa AbstractState && (s.t += 1)
    end
    return nothing
end

update!(::NamedTuple{(), Tuple{}}, ::Any, ::Any, opt::AbstractOptimizer) = nothing
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


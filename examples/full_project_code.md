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
using Zygote, Tullio
using ChainRulesCore, ForwardDiff
using Printf, TimerOutputs
using Metal, LoopVectorization
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

include("fileIO/utils.jl")
include("fileIO/mnist.jl")

include("algorithm/train.jl")

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

abstract type AbstractNanoTensor{T, N} <: AbstractArray{T, N} end
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
    x_reshaped = reshape(train_x_raw, 1, 28, 28, :)
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
    data::A # (Channel, D1, D2, ... , Batch)
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
    out_idx = [Symbol("x_$d") for d in 1:N] # 输出索引 x_1
    red_idx = [Symbol("i_$d") for d in 1:N] # 归约索引 i_1
    
    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("k_$d")) = l.k[$d]))
    end
    
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        
        s_node = Expr(:$, s_sym) 
        
        sub_node = :($(out_idx[d]) - 1)
        
        mult_node = Expr(:call, :*, sub_node, s_node)
        
        final_node = Expr(:call, :+, mult_node, red_idx[d])
        
        return final_node
    end
    
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - $(Symbol("k_$d"))) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]
    
    ranges_list = []
    for d in 1:N
        push!(ranges_list, :($(out_idx[d]) in 1:$(Symbol("out_dim_$d"))))
        push!(ranges_list, :($(red_idx[d]) in 1:$(Symbol("k_$d"))))
    end
    ranges_tuple = Expr(:tuple, ranges_list...) # 拼成 (r1, r2...)

    @eval begin
        # Mean Pooling
        function (l::Pool{$N})(::typeof(mean), x::SpatialTensor{$N})
            X = x.data
            in_size = size(X)
            $(unpack_exprs...)
            $(size_calc_exprs...)
            
            # @tullio y[c, $(out_idx...), b] := mean(X[c, $(access_exprs...), b]) $ranges_tuple
            @tullio y[c, $(out_idx...), b] := mean(X[c, $(access_exprs...), b]) $ranges_tuple grad=Dual

            return SpatialTensor{$N}(y)
        end

        # Max Pooling
        function (l::Pool{$N})(::typeof(maximum), x::SpatialTensor{$N})
            X = x.data
            in_size = size(X)
            $(unpack_exprs...)
            $(size_calc_exprs...)
            
            # @tullio y[c, $(out_idx...), b] := maximum(X[c, $(access_exprs...), b]) $ranges_tuple
            @tullio y[c, $(out_idx...), b] := maximum(X[c, $(access_exprs...), b]) $ranges_tuple grad=Dual
            
            return SpatialTensor{$N}(y)
        end
    end
end

function (l::Pool)(x::SpatialTensor)
    return l(l.mode, x)
end



```

---

## File: src/module/utils.jl
```julia

function _fmt_shape(s)
    dims = s[1:end-1] 
    return string(dims)
end

_count_params(l::Any) = 0
_count_params(l::Dense) = length(l.W) + length(l.b)
_count_params(l::Conv)  = length(l.W) + length(l.b)

function format_number(n::Int)
    return replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
end
```

---

## File: src/module/sequential.jl
```julia
struct Sequential <: AbstractModule
    layers::Vector{AbstractModule}
    function Sequential(A::AbstractVector)
        S = new(convert(Vector{AbstractModule},A))
        _check(S)
        return S
    end
end
Sequential(layers...) = Sequential(collect(layers))

Base.iterate(S::Sequential) = iterate(S.layers)
Base.iterate(S::Sequential, state) = iterate(S.layers, state)
Base.length(S::Sequential) = length(S.layers)
Base.getindex(S::Sequential, i) = getindex(S.layers, i)
Base.lastindex(S::Sequential) = lastindex(S.layers)
Base.eltype(::Type{Sequential}) = AbstractModule

function (model::Sequential)(x)
    for layer in model.layers
        x = layer(x)
    end
    return x
end

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

(l::Input)(x) = x

_count_params(l::Input) = 0
```

---

## File: src/module/dense.jl
```julia

struct Dense{TW, TB, F} <: AbstractModule
    W::TW  # (Out_Dim, In_Dim)
    b::TB  # (Out_Dim)
    act::F
end

function Dense(in_dim::Int, out_dim::Int, act=identity)

    scale = sqrt(2.0f0 / in_dim)
    W = randn(Float32, out_dim, in_dim) .* scale
    

    b = zeros(Float32, out_dim)
    
    return Dense{typeof(W), typeof(b), typeof(act)}(W, b, act)
end


function (l::Dense)(x::FlatTensor)

    y_linear = l.W * x.data
    y_pre_act = y_linear .+ l.b
    
    y = l.act.(y_pre_act)

    return FlatTensor(y)
end

function (l::Dense)(x::SpatialTensor)
    raw_data = x.data
    full_size = size(raw_data)
    batch_size = full_size[end]
    
    flat_features = length(raw_data) ÷ batch_size
    
    expected_in = size(l.W, 2)
    
    if flat_features != expected_in
        error(b"❌ Auto-Flatten Dimension Mismatch!\n" *
              "Dense layer expects input dim: $(expected_in)\n" *
              "But incoming tensor $(full_size) flattens to: $(flat_features)\n" *
              "Check your Conv/Pool parameters.")
    end

    flat_data = reshape(raw_data, flat_features, batch_size)

    return l(FlatTensor(flat_data))
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


function (::Flatten)(x::SpatialTensor{D}) where D
    batch_size = size(x.data)[end]
    flat_data = reshape(x.data, :, batch_size)
    return FlatTensor(flat_data)
end

function (::Flatten)(x::FlatTensor)
    return x
end
```

---

## File: src/module/check.jl
```julia
using Printf
using Random

"""
    _check(model::Sequential, input_shape::Tuple)

运行一次虚拟前向传播，检查层维度匹配情况，并打印详细摘要。
input_shape: (Channel, H, W) 或 (Channel, Len) 等，不包含 Batch 维度。
"""
function _check(layers::Vector{AbstractModule},input_shape::Union{Tuple, Nothing}=nothing)
    println("="^80)
    println("Model Architecture Inspector")
    println("="^80)

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

    full_shape = (input_shape..., 1)
    x_data = randn(Float32, full_shape)
    
    x = SpatialTensor{spatial_dims}(x_data)
    
    println(@sprintf("Input Signal: %s (Batch=1)", string(size(x))))
    println("-"^80)
    @printf("%-4s %-15s %-25s %-25s %-10s\n", "ID", "Layer Type", "Input Shape", "Output Shape", "Params")
    println("-"^80)

    total_params = 0
    
    for (i, layer) in enumerate(layers)
        layer_type = string(typeof(layer))
        layer_name = split(layer_type, "{")[1]
        in_shape = size(x)
        
        try
            out = layer(x)
            
            out_shape = size(out)
            
            n_params = _count_params(layer)
            total_params += n_params
            
            str_in  = _fmt_shape(in_shape)
            str_out = _fmt_shape(out_shape)
            
            @printf("%-4d %-15s %-25s %-25s %-10d\n", 
                    i, layer_name, str_in, str_out, n_params)
            
            x = out
            
        catch e
            println("\n" * "!"^80)
            println("Layer Dimension Mismatch Detected at Layer $i [$layer_name]!")
            println("!"^80)
            println("   Expected Input: Compatible with $(_fmt_shape(in_shape))")
            
            if layer isa Dense
                println("   Layer Config:   InputDim = $(size(layer.W, 2))")
                println("   Analysis:       The Dense layer expects $(size(layer.W, 2)) features, but received $(in_shape[1]).")
                println("                   (Did you calculate the Flatten output size correctly?)")
            elseif layer isa Conv
                println("   Analysis:       Convolution failure. Check if input spatial size is smaller than Kernel size.")
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

_check(model::Sequential) = _check(model.layers)
```

---

## File: src/module/convolution.jl
```julia

struct Conv{D, TW, TB, F} <: AbstractModule
    W::TW
    b::TB
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    act::F
end

function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; stride=1, dilation=1, act=identity)

    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    
    w_shape = (out_ch, in_ch, ks...)

    fan_in = in_ch * prod(ks)
    scale = sqrt(2.0 / fan_in)
    W = randn(Float32, w_shape...) .* Float32(scale)
    b = zeros(Float32, out_ch)
    
    return Conv{D, typeof(W), typeof(b), typeof(act)}(W, b, st, di, act)
end

for N in 1:3
    input_idxs = [Symbol("x_$d") for d in 1:N] # 输出坐标: x_1, x_2
    kern_idxs  = [Symbol("k_$d") for d in 1:N] # 核坐标: k_1, k_2

    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("d_$d")) = l.dilation[$d]))
        push!(unpack_exprs, :($(Symbol("ksize_$d")) = size(l.W, $d + 2)))
    end
    
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        d_sym = Symbol("d_$d")

        :(( $(input_idxs[d]) - 1 ) * $(Expr(:$, s_sym)) + ( $(kern_idxs[d]) - 1 ) * $(Expr(:$, d_sym)) + 1)
    end

    # H_out = (H_in - dilation*(k-1) - 1) ÷ stride + 1
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - ($(Symbol("d_$d")) * ($(Symbol("ksize_$d")) - 1) + 1)) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]

    # (x_1 in 1:out_dim_1, x_2 in 1:out_dim_2)
    ranges = Expr(:tuple, [ :($(input_idxs[d]) in 1:$(Symbol("out_dim_$d"))) for d in 1:N ]...)

    left_side = :(y[o, $(input_idxs...), b])
    
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N})
            X = x.data
            W = l.W
            b_vec = l.b
            in_size = size(X)
            
            $(unpack_exprs...)
            $(size_calc_exprs...)
            @tullio $left_side := W[o, c, $(kern_idxs...)] * X[c, $(access_exprs...), b] $ranges
            
            bias_shape = (length(b_vec), $(ones(Int, N)...), 1)
            y = y .+ reshape(b_vec, bias_shape)
            
            return SpatialTensor{$N}(l.act.(y))
        end
    end
end


```

---

## File: src/algorithm/train.jl
```julia

function train!(model::AbstractModule, train_loader, algo::SimpleAlgorithm)

    history = TrainingHistory()
    velocities = _initial_velocities(model)
    manualGC()

    total_loaders = length(train_loader)

    for epoch in 1:algo.epochs
        for (x_raw, y_raw) in train_loader

            @timeit TO "Data Prepare" begin
                ndims_spatial = ndims(x_raw) - 2
                x = SpatialTensor{ndims_spatial}(x_raw)
                y = y_raw
            end

            @timeit TO "Back Propagation" begin
                loss_val = _train_step!(model, x, y, algo, velocities, history)
            end

            if algo.show_times > 0 && mod(mod(history.count - 1, total_loaders) + 1, algo.show_times) == 0
                print("Epoch $(epoch) [$(mod(history.count-1, total_loaders) + 1)/$(total_loaders)] - ")
                show(history)
            end

            history.count += 1
        end 
        history.avg_loss = mean(history.loss[end - length(train_loader) + 1:end])
        history.avg_acc = mean(history.accuracy[end - length(train_loader) + 1:end])

        @timeit TO "gc" manualGC()

        show(TO;title = "$(epoch) / $(algo.epochs)")
        print("\n")
        show(history)

        if algo.target_loss !== nothing && history.avg_loss  <= algo.target_loss
            if history.count_loss ≥ algo.patience
                println()
                println(bg"Target Loss Reached!"," ($(history.avg_loss) <= $(algo.target_loss))")
                println("Stopping training early at Epoch $epoch.")
                break
            else
                history.count_loss += 1
            end
        end

        if algo.target_acc !== nothing && history.avg_acc >= algo.target_acc
            if history.count_acc ≥ algo.patience
                println()
                println(bg"Target Accuracy Reached!"," ($(history.avg_acc) >= $(algo.target_acc))")
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


function _train_step!(model::AbstractModule, x, y, 
                      algo::SimpleAlgorithm, 
                      velocities::IdDict, 
                      history::TrainingHistory)
    
    ps = _params(model)

    loss_val, gs = @timeit TO "calc gradient" Zygote.withgradient(ps) do
        loss(model, x, y)
    end

    @timeit TO "update" for p in ps
        if gs[p] !== nothing
            g = gs[p]

            v = get!(velocities, p, zeros(eltype(p), size(p)))

            @. v = algo.momentum * v + g
            @. p -= algo.learning_rate * v
        end
    end

    if isdefined(history, :loss)
        push!(history.loss, loss_val)
        push!(history.accuracy, accuracy(model, x, y))
    end
    
    return loss_val
end

function _initial_velocities(model::AbstractModule)
    velocities = IdDict()
    ps = _params(model)
    for p in ps
        velocities[p] = zeros(eltype(p), size(p))
    end
    return velocities
end

function loss(model::AbstractModule, x, y)
    y_pred = model(x)
    logits = y_pred.data
    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    return -sum(y .* log.(probs .+ 1e-10)) / size(logits, 2)
end

function accuracy(model::AbstractModule, x, y)
    y_pred = model(x)
    logits = y_pred.data
    pred_idx = [c[1] for c in argmax(logits, dims=1)]
    true_idx = [c[1] for c in argmax(y, dims=1)]
    return mean(pred_idx .== true_idx)
end

"""
    params(m::AbstractModule)

返回一个 Zygote.Params 对象，包含该模块及其子模块的所有可训练参数。
"""
function _params(m::AbstractModule)
    ps = Params()
    _collect_params!(ps, m)
    return ps
end

function _collect_params!(ps::Params, m::Sequential)
    for layer in m.layers
        _collect_params!(ps, layer)
    end
end

function _collect_params!(ps::Params, m::AbstractModule)
    if hasfield(typeof(m), :W)
        push!(ps, m.W)
    end
    if hasfield(typeof(m), :b)
        push!(ps, m.b)
    end
    # 其他参数
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
@kwdef struct SimpleAlgorithm <: AbstractAlgorithm
    learning_rate::Float32               = 1e-2
    momentum::Float32                    = 0.9f0
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


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
    
    # 1. 加载原始数据
    # train_x_raw: (28, 28, 60000) :: Matrix{Gray{N0f8}} 或 Float32
    # train_y_raw: (60000,) :: Int64
    train_x_raw, train_y_raw = MNIST(split=:train)[:]
    
    # 2. 预处理 X (Features)
    # 原始形状: (Width, Height, Batch) -> (28, 28, 60000)
    # 目标形状: (Channel, Height, Width, Batch) -> (1, 28, 28, 60000)
    # -----------------------------------------------------------
    # [关键修改]
    # 我们不再 flatten 成 784。
    # 而是插入 Channel=1 的维度，适配 SpatialTensor{2}。
    # -----------------------------------------------------------
    x_reshaped = reshape(train_x_raw, 1, 28, 28, :)
    x_data = Float32.(x_reshaped) # 强烈建议用 Float32
    
    # 3. 预处理 Y (Labels)
    # 目标形状: (Classes, Batch) -> (10, 60000)
    # 使用 One-Hot 编码
    y_oh = onehotbatch(train_y_raw, 0:9)
    y_data = Float32.(y_oh) # 标签也转为 Float32，方便计算 Loss
    
    # 4. 构建 DataLoader
    # MLUtils 会自动沿着最后一个维度 (dim=4 for x, dim=2 for y) 切片
    # 每次迭代返回: 
    #   x_batch: (1, 28, 28, batch_size)
    #   y_batch: (10, batch_size)
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
    data::A
    
    # 内部构造函数：强制检查维度约束
    function SpatialTensor{D}(data::A) where {D, A}
        @assert ndims(A) == D + 2 "SpatialTensor{$D} 需要 $(D+2) 维数据: (Channel, Spatial..., Batch)"
        new{D, eltype(A), ndims(A), A}(data)
    end
end




# 定义 FlatTensor 用于 Dense 层输入
struct FlatTensor{T, A<:AbstractArray{T, 2}} <: AbstractNanoTensor{T, 2}
    data::A # (Features, Batch)
end



```

---

## File: src/wrapper/interface.jl
```julia
# ==========================================
# 1. 基础数组接口 (继承自 AbstractNanoTensor)
# ==========================================
# 如果你之前运行过 AbstractNanoTensor 的通用代码，这一部分其实是复用的。
# 为了保险起见，这里列出 FlatTensor 依赖的逻辑：

# 确保 FlatTensor 表现得像一个矩阵
Base.size(t::FlatTensor) = size(t.data)
Base.size(t::FlatTensor, d::Int) = size(t.data, d)
Base.length(t::FlatTensor) = length(t.data)
Base.getindex(t::FlatTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::FlatTensor, v, i...) = setindex!(t.data, v, i...)
Base.IndexStyle(::Type{<:FlatTensor}) = IndexLinear()
Base.eltype(::FlatTensor{T}) where T = T

# ==========================================
# 2. 打印美化 (Show) - FlatTensor 特化版
# ==========================================
# 让 REPL 打印出 "FlatTensor(Float32, 128×4)" 而不是乱七八糟的结构体信息

function Base.show(io::IO, t::FlatTensor)
    # 获取维度 (Features, Batch)
    dims = size(t)
    # 打印简略信息
    print(io, "FlatTensor($(eltype(t)), $(join(dims, "×")))")
end

# 当在 REPL 直接输入变量回车时的详细打印
function Base.show(io::IO, ::MIME"text/plain", t::FlatTensor)
    summary(io, t) # 打印头部信息
    println(io, ":")
    # 调用 Base 的数组打印逻辑，显示实际数值
    Base.print_array(io, t.data)
end

# 自定义 summary，配合上面的 show 使用
Base.summary(io::IO, t::FlatTensor) = print(io, "FlatTensor($(eltype(t)), $(join(size(t), "×")))")


# ==========================================
# 3. 广播支持 (Broadcasting) - 关键！
# ==========================================
# 这一步保证了当你做 relu.(flat_tensor) 时，
# 结果依然是 FlatTensor，而不是退化成普通的 Matrix。

# 只有当结果依然是 2D 时，才包装回 FlatTensor
function Base.similar(t::FlatTensor, ::Type{T}, dims::Dims) where {T}
    if length(dims) == 2
        # 保持 FlatTensor 包装
        return FlatTensor(similar(t.data, T, dims))
    else
        # 如果维度变了 (比如 sum 变成了 1D)，则退化为普通 Array
        return similar(t.data, T, dims)
    end
end

# ==========================================
# AbstractNanoTensor 基础接口实现
# ==========================================

# 1. 核心接口: size
# 解决报错: MethodError: no method matching size(...)
Base.size(t::AbstractNanoTensor) = size(t.data)
Base.size(t::AbstractNanoTensor, d::Int) = size(t.data, d)
Base.length(t::AbstractNanoTensor) = length(t.data)

# 2. 核心接口: getindex / setindex!
# 让你能像操作普通数组一样操作 Tensor: x[1, 2], x[1] = 5
Base.getindex(t::AbstractNanoTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::AbstractNanoTensor, v, i...) = setindex!(t.data, v, i...)

# 3. 性能优化: IndexStyle
# 告诉 Julia 你的底层数据是线性存储的 (Array/CuArray 都是)，这样迭代更快
Base.IndexStyle(::Type{<:AbstractNanoTensor}) = IndexLinear()

# 4. 类型推导: eltype
# 让 Julia 知道里面存的是 Float32 还是 Float64
Base.eltype(::AbstractNanoTensor{T}) where T = T

# ==========================================
# 打印美化 (Show) - Debug 神器
# ==========================================

# 让 REPL 打印出来更直观，而不是显示一大堆数据
function Base.show(io::IO, t::SpatialTensor{D}) where D
    dims = size(t)
    # 打印格式: SpatialTensor{2}(Float32, 3×28×28×4)
    print(io, "SpatialTensor{$D}($(eltype(t)), $(join(dims, "×")))")
end

function Base.show(io::IO, t::FlatTensor)
    dims = size(t)
    print(io, "FlatTensor($(eltype(t)), $(join(dims, "×")))")
end

# 针对 MIME (比如在 Jupyter Notebook 显示)
function Base.show(io::IO, ::MIME"text/plain", t::AbstractNanoTensor)
    print(io, typeof(t), " with size ", size(t), ":\n")
    # 调用底层数组的打印逻辑，只显示一部分数据，避免刷屏
    Base.print_array(io, t.data)
end

# ==========================================
# 广播支持 (Broadcasting) - 进阶
# ==========================================
# 这是一个简化的广播实现。
# 当你做 t + 1 时，默认行为会返回一个普通 Array。
# 如果你希望它尽可能保持 Tensor 包装，可以加上类似的适配器 (可选):

Base.similar(t::SpatialTensor{D}, ::Type{T}, dims::Dims) where {D, T} = 
    SpatialTensor{D}(similar(t.data, T, dims))

Base.similar(t::FlatTensor, ::Type{T}, dims::Dims) where {T} = 
    FlatTensor(similar(t.data, T, dims))
```

---

## File: src/module/pool.jl
```julia

# ============================
# 1. 结构体定义
# ============================
struct Pool{D, F} <: AbstractModule
    k::NTuple{D, Int}
    stride::NTuple{D, Int}
    mode::F # mean 或 maximum
end

# 智能构造函数
function Pool(D::Int, k::Union{Int, NTuple}; 
              stride=k, mode=mean) # 默认 stride=k (不重叠)
    
    ks = k isa Int ? ntuple(_->k, D) : k
    st = stride isa Int ? ntuple(_->stride, D) : stride
    
    return Pool{D, typeof(mode)}(ks, st, mode)
end

for N in 1:3
    # A. 准备符号
    out_idx = [Symbol("x_$d") for d in 1:N] # 输出索引 x_1
    red_idx = [Symbol("i_$d") for d in 1:N] # 归约索引 i_1
    
    # B. 准备解包代码 (生成 s_1 = l.stride[1] ...)
    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("k_$d")) = l.k[$d]))
    end
    
    # C. 手动构建坐标公式 (避免 $ 语法错误)
    # 目标公式: (x_1 - 1) * $s_1 + i_1
    # 我们用 Expr(:call, ...) 手动拼装，而不是写在 :(...) 里
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        
        # 1. 构造 "$s_1" 这个节点
        s_node = Expr(:$, s_sym) 
        
        # 2. 构造 (x_1 - 1)
        sub_node = :($(out_idx[d]) - 1)
        
        # 3. 构造 (x - 1) * $s
        mult_node = Expr(:call, :*, sub_node, s_node)
        
        # 4. 构造 ... + i_1
        final_node = Expr(:call, :+, mult_node, red_idx[d])
        
        return final_node
    end
    
    # D. 构造输出尺寸计算代码
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - $(Symbol("k_$d"))) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]
    
    # E. 构造循环范围
    # 注意: Tullio 的范围不需要 $ (如 i in 1:k_1 即可)，只需要公式里有 $
    ranges_list = []
    for d in 1:N
        # 输出范围: x_1 in 1:out_dim_1
        push!(ranges_list, :($(out_idx[d]) in 1:$(Symbol("out_dim_$d"))))
        # 归约范围: i_1 in 1:k_1
        push!(ranges_list, :($(red_idx[d]) in 1:$(Symbol("k_$d"))))
    end
    ranges_tuple = Expr(:tuple, ranges_list...) # 拼成 (r1, r2...)

    # F. 生成函数
    @eval begin
        # Mean Pooling
        function (l::Pool{$N})(::typeof(mean), x::SpatialTensor{$N})
            X = x.data
            in_size = size(X)
            $(unpack_exprs...)
            $(size_calc_exprs...)
            
            # 手动拼接 @tullio 调用
            # 注意：access_exprs 里面已经包含了我们手动构造的 $ 节点
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

# ============================
# 3. 统一入口
# ============================
function (l::Pool)(x::SpatialTensor)
    return l(l.mode, x)
end

# _params(::Params, ::Pooling) = nothing


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

# 前向传播：直接返回 x，不做任何操作，零开销
(l::Input)(x) = x

# 辅助函数：如果是 Input 层，不算参数
_count_params(l::Input) = 0
```

---

## File: src/module/dense.jl
```julia
# ============================
# 1. 结构体定义
# ============================
# TW: 权重类型 (Matrix)
# TB: 偏置类型 (Vector)
# F:  激活函数类型
struct Dense{TW, TB, F} <: AbstractModule
    W::TW  # (Out_Dim, In_Dim)
    b::TB  # (Out_Dim)
    act::F
end

# ============================
# 2. 构造函数
# ============================
function Dense(in_dim::Int, out_dim::Int, act=identity)
    # 初始化权重 (He Initialization / Kaiming Init)
    # 适用于 ReLU 等非线性激活
    scale = sqrt(2.0f0 / in_dim)
    W = randn(Float32, out_dim, in_dim) .* scale
    
    # 初始化偏置 (通常全0)
    b = zeros(Float32, out_dim)
    
    return Dense{typeof(W), typeof(b), typeof(act)}(W, b, act)
end

# ============================
# 3. 前向传播 (Forward)
# ============================
# 核心约束：只接受 FlatTensor
function (l::Dense)(x::FlatTensor)
    # x.data 维度: (In_Dim, Batch)
    # l.W    维度: (Out_Dim, In_Dim)
    
    # 1. 线性变换 (Matrix Multiplication)
    # (Out, In) * (In, Batch) -> (Out, Batch)
    y_linear = l.W * x.data
    
    # 2. 加上偏置 (Bias Broadcasting)
    # Julia 的 .+ 会自动把 b (Out_Dim) 广播到 (Out_Dim, Batch)
    y_pre_act = y_linear .+ l.b
    
    # 3. 激活函数
    y = l.act.(y_pre_act)
    
    # 4. 保持类型流转：输出依然是 FlatTensor
    # 这样下一个 Dense 层可以直接接收它
    return FlatTensor(y)
end

function (l::Dense)(x::SpatialTensor)
    # 1. 获取原始数据 (C, H, W, B)
    raw_data = x.data
    full_size = size(raw_data)
    batch_size = full_size[end]
    
    # 2. 计算展平后的特征数
    # 总元素数 除以 Batch数 = 单个样本的特征数
    # 例如 (16, 4, 4, 10) -> 总共 2560 -> 除以10 -> 特征数 256
    flat_features = length(raw_data) ÷ batch_size
    
    # 3. 安全检查 (这一点非常重要！)
    # 确保展平后的维度能和 Dense 的权重矩阵 W 对得上
    expected_in = size(l.W, 2)
    
    if flat_features != expected_in
        # 友好的报错信息
        error(b"❌ Auto-Flatten Dimension Mismatch!\n" *
              "Dense layer expects input dim: $(expected_in)\n" *
              "But incoming tensor $(full_size) flattens to: $(flat_features)\n" *
              "Check your Conv/Pool parameters.")
    end
    
    # 4. 执行 Reshape (Julia 的 reshape 通常是零拷贝的，非常快)
    # 变成 (Features, Batch)
    flat_data = reshape(raw_data, flat_features, batch_size)
    
    # 5. 递归调用
    # 把它包装成 FlatTensor，扔给上面的标准路径去算
    return l(FlatTensor(flat_data))
end
```

---

## File: src/module/flatten.jl
```julia
# ============================
# 1. 结构体定义
# ============================
"""
    Flatten()

将任意维度的 SpatialTensor 展平为 FlatTensor。
输入: (Channel, D1, D2..., Batch)
输出: (Features, Batch)
"""
struct Flatten <: AbstractModule end

# ============================
# 2. 通用 Forward 实现
# ============================
# 这里的 {D} 泛型让它自动适配 1D/2D/3D 的 SpatialTensor
function (::Flatten)(x::SpatialTensor{D}) where D
    # x.data 的维度: (Channel, Spatial_1, ..., Spatial_D, Batch)
    
    # 获取 Batch 大小 (也就是最后一个维度)
    batch_size = size(x.data)[end]
    
    # 核心魔法: reshape(data, :, batch_size)
    # ":" (Colon) 会自动计算前面所有维度的乘积 (C * S1 * ... * SD)
    # 在 Julia 中，这是一个极快的操作
    flat_data = reshape(x.data, :, batch_size)
    
    # 包装成 Dense 层需要的类型
    return FlatTensor(flat_data)
end

# (可选) 容错处理: 如果已经是 FlatTensor，直接返回，不做操作
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


    # 如果用户没传 shape，尝试从第一层读取
    if input_shape === nothing
        if layers[1] isa Input
            input_shape = layers[1].shape
            # println("ℹ️  Auto-detected input shape from layer 1: $input_shape")
            # 真正的计算从第2层开始，因为第1层是虚拟的
            # 但为了显示好看，我们依然遍历它
        else
            error("Missing input_shape! \nPlease provide it as an argument OR add an Input(shape) layer at the start of your model.")
        end
    end
    
    # 1. 构造 Dummy Input (Batch Size = 1)
    # ----------------------------------------------------
    # 根据 input_shape 的长度自动判断是 SpatialTensor{1}, {2} 还是 {3}
    spatial_dims = length(input_shape) - 1 # 减去 Channel 维
    if spatial_dims < 1
        error("Input shape must be at least (Channel, Len...), got $input_shape")
    end
    
    # 构造数据 (Channel, D1, D2..., Batch=1)
    full_shape = (input_shape..., 1)
    x_data = randn(Float32, full_shape)
    
    # 包装成 Tensor
    x = SpatialTensor{spatial_dims}(x_data)
    
    println(@sprintf("Input Signal: %s (Batch=1)", string(size(x))))
    println("-"^80)
    @printf("%-4s %-15s %-25s %-25s %-10s\n", "ID", "Layer Type", "Input Shape", "Output Shape", "Params")
    println("-"^80)

    total_params = 0
    
    # 2. 逐层运行
    # ----------------------------------------------------
    for (i, layer) in enumerate(layers)
        layer_type = string(typeof(layer))
        # 只保留 Struct 名字，去掉 {...}
        layer_name = split(layer_type, "{")[1]
        
        # 记录输入形状
        in_shape = size(x)
        
        # 尝试运行该层
        try
            # --- FORWARD PASS ---
            out = layer(x)
            # --------------------
            
            out_shape = size(out)
            
            # 计算参数量
            n_params = _count_params(layer)
            total_params += n_params
            
            # 格式化打印
            str_in  = _fmt_shape(in_shape)
            str_out = _fmt_shape(out_shape)
            
            @printf("%-4d %-15s %-25s %-25s %-10d\n", 
                    i, layer_name, str_in, str_out, n_params)
            
            # 更新 x 为下一层的输入
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
            return # 终止检查
        end
    end
    
    println("-"^80)
    println(g"CHECK PASSED")
    println("Total Parameters: $(format_number(total_params))")
    println("="^80)
end

# ---------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------

# 格式化形状字符串 (去除 Batch=1)
function _fmt_shape(s)
    # s 是 (C, H, W, B) 或 (Features, B)
    # 我们显示时不显示 Batch，显得更干净
    dims = s[1:end-1] 
    return string(dims)
end

# 计算参数量
_count_params(l::Any) = 0
_count_params(l::Dense) = length(l.W) + length(l.b)
_count_params(l::Conv)  = length(l.W) + length(l.b)

# 格式化数字 (1,234,567)
function format_number(n::Int)
    return replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
end


_check(model::Sequential) = _check(model.layers)
```

---

## File: src/module/convolution.jl
```julia
# D: 空间维度
# TW: 权重类型 (Tensor Weight) -> 4D/5D Array
# TB: 偏置类型 (Tensor Bias)   -> 1D Vector
# F: 激活函数
struct Conv{D, TW, TB, F} <: AbstractModule
    W::TW
    b::TB
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    act::F
end
# 智能构造函数
function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; stride=1, dilation=1, act=identity)
    
    # 1. 规范化参数为 NTuple
    # 如果用户只传了 k=3，自动变成 (3, 3, ...)
    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    
    # 2. 初始化权重
    # 权重形状: (Out, In, k_1, k_2, ..., k_D)
    w_shape = (out_ch, in_ch, ks...)
    
    # He Initialization
    fan_in = in_ch * prod(ks)
    scale = sqrt(2.0 / fan_in)
    W = randn(Float32, w_shape...) .* Float32(scale)
    b = zeros(Float32, out_ch)
    
    return Conv{D, typeof(W), typeof(b), typeof(act)}(W, b, st, di, act)
end

for N in 1:3
    # A. 准备符号
    input_idxs = [Symbol("x_$d") for d in 1:N] # 输出坐标: x_1, x_2
    kern_idxs  = [Symbol("k_$d") for d in 1:N] # 核坐标: k_1, k_2
    
    # B. 准备参数解包代码 (生成 s_1 = l.stride[1] 等)
    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("d_$d")) = l.dilation[$d]))
        push!(unpack_exprs, :($(Symbol("ksize_$d")) = size(l.W, $d + 2)))
    end
    
    # C. 构造索引公式 (关键修复！)
    # 我们需要生成的代码类似于: (x_1 - 1) * $s_1 + ...
    # 在元编程中，要生成 '$s_1'，需要写成 Expr(:$, :s_1)
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        d_sym = Symbol("d_$d")
        
        # 构造: (x - 1) * $s + (k - 1) * $d + 1
        # 注意这里的 Expr(:$, s_sym) 是为了让 Tullio 看到 $s_1
        :(( $(input_idxs[d]) - 1 ) * $(Expr(:$, s_sym)) + ( $(kern_idxs[d]) - 1 ) * $(Expr(:$, d_sym)) + 1)
    end
    
    # D. 构造输出尺寸计算代码
    # H_out = (H_in - dilation*(k-1) - 1) ÷ stride + 1
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - ($(Symbol("d_$d")) * ($(Symbol("ksize_$d")) - 1) + 1)) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]
    
    # E. 构造循环范围 Tuple
    # (x_1 in 1:out_dim_1, x_2 in 1:out_dim_2)
    ranges = Expr(:tuple, [ :($(input_idxs[d]) in 1:$(Symbol("out_dim_$d"))) for d in 1:N ]...)

    # F. 构造 Tullio 左值
    left_side = :(y[o, $(input_idxs...), b])
    
    # --- 生成最终函数 ---
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N})
            X = x.data
            W = l.W
            b_vec = l.b
            
            # 1. 获取输入尺寸 (注意: index 1 是 channel，所以空间维从 2 开始)
            in_size = size(X)
            
            # 2. 解包 stride, dilation (s_1, d_1...)
            $(unpack_exprs...)
            
            # 3. 计算输出尺寸 (out_dim_1...)
            $(size_calc_exprs...)
            
            # 4. 核心计算 (已插入 $ 符号)
            @tullio $left_side := W[o, c, $(kern_idxs...)] * X[c, $(access_exprs...), b] $ranges
            
            # 5. Bias 广播
            # 形状: (Out, 1..., 1)
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
    # 1. 初始化基础设施
    history = TrainingHistory()

    # 3. 初始化动量缓存 (Velocities)
    # 这一步也需要适配 AbstractModule，下面有定义
    velocities = _initial_velocities(model)
    
    # 4. 垃圾回收 (清理之前的内存)
    manualGC()

    total_batches = try length(train_loader) catch; 0 end
    for epoch in 1:algo.epochs
        count = 1
        # 进度打印辅助
        for (x_raw, y_raw) in train_loader
            
            # --- A. 数据准备 ---
            @timeit TO "Data Prepare" begin
                # 自动识别维度并包装 (适配 Conv 和 Dense)
                ndims_spatial = ndims(x_raw) - 2
                x = SpatialTensor{ndims_spatial}(x_raw)
                y = y_raw
            end

            # --- B. 核心训练 (Forward + Backward + Update) ---
            @timeit TO "Back Propagation" begin
                loss_val = _train_step!(model, x, y, algo, velocities, history)
            end

            # --- C. 日志打印 ---
            if algo.show_times > 0 && mod(count, algo.show_times) == 0
                print("Epoch $(epoch) [$(count)/$(total_batches)] - ")
                show(history)
            end
            count += 1
        end        
        @timeit TO "gc" manualGC()

        show(TO;title = "$(epoch) / $(algo.epochs)")
        print("\n")
        show(history)
    end
    
    show(TO)
    
    return history
end

# ==============================================================================
# 3. 单步训练 (Step)
# ==============================================================================

function _train_step!(model::AbstractModule, x, y, 
                      algo::SimpleAlgorithm, 
                      velocities::IdDict, 
                      history::TrainingHistory)
    
    # 1. 获取参数 (使用通用的 params 接口)
    # Zygote 会智能处理，这一步开销很小
    ps = _params(model)
    
    # 2. 计算梯度
    loss_val, gs = @timeit TO "calc gradient" Zygote.withgradient(ps) do
        loss(model, x, y)
    end
    
    # 3. 参数更新 (Momentum SGD)
    # 注意：这里我们遍历的是 ps (参数集合)，而不是 layers
    # 这意味着无论网络结构多复杂，更新逻辑都是统一的
    @timeit TO "update" for p in ps
        if gs[p] !== nothing
            g = gs[p]
            
            # 从缓存中取出对应的速度，如果没有就初始化为0
            # (这是防止 _initial_velocities 漏掉某些参数的安全网)
            v = get!(velocities, p, zeros(eltype(p), size(p)))
            
            # 动量更新 (In-place)
            @. v = algo.momentum * v + g
            @. p -= algo.learning_rate * v
        end
    end

    # 4. 记录日志
    if isdefined(history, :loss)
        push!(history.loss, loss_val)
        push!(history.accuracy, accuracy(model, x, y))
    end
    
    return loss_val
end

# ==============================================================================
# 4. 辅助函数：初始化动量
# ==============================================================================

function _initial_velocities(model::AbstractModule)
    velocities = IdDict()
    ps = _params(model) # 复用通用的 params
    for p in ps
        velocities[p] = zeros(eltype(p), size(p))
    end
    return velocities
end

# ==============================================================================
# 5. Loss 和 Accuracy (保持不变，但签名改为 AbstractModule)
# ==============================================================================

function loss(model::AbstractModule, x, y)
    y_pred = model(x) # 调用 AbstractModule 的前向传播
    # ... (具体的 CrossEntropy 逻辑同前) ...
    # 为了完整性简写如下：
    logits = y_pred.data
    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    return -sum(y .* log.(probs .+ 1e-10)) / size(logits, 2)
end

function accuracy(model::AbstractModule, x, y)
    y_pred = model(x)
    logits = y_pred.data
    pred_idx = [c[1] for c in argmax(logits, dims=1)]
    # 假设 y 是 One-Hot
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
    # 检查是否有 W 字段
    if hasfield(typeof(m), :W)
        push!(ps, m.W)
    end
    # 检查是否有 b 字段
    if hasfield(typeof(m), :b)
        push!(ps, m.b)
    end
    # 如果未来有其他参数 (比如 Gamma, Beta)，也可以加在这里
end
```

---

## File: src/control/algorithm.jl
```julia
# struct SimpleAlgorithm <: AbstractAlgorithm
#     batch_size::Int64
#     epochs::Int64
#     learning_rate::Float64
#     momentum::Float64
#     show_times::Number
# end
"""
    SimpleAlgorithm(args...)

配置训练的超参数。
支持关键字构建，例如: `SimpleAlgorithm(learning_rate=1e-3, epochs=20)`
"""
@kwdef struct SimpleAlgorithm <: AbstractAlgorithm
    learning_rate::Float32 = 1e-2
    momentum::Float32      = 0.9f0
    epochs::Int            = 10
    batch_size::Int        = 32
    show_times::Int        = 1
end
```

---

## File: src/control/information.jl
```julia

# 1. 改名为 TrainingHistory，只存指标
mutable struct TrainingHistory <: AbstractInformation
    loss::Vector{Float64}
    accuracy::Vector{Float64}
    # velocities 被移除了，因为它属于训练过程中的临时状态
    
    TrainingHistory() = new(Float64[], Float64[])
end

# 2. 更安全的 show 函数
function Base.show(io::IO, h::TrainingHistory)
    if isempty(h.loss)
        print(io, "TrainingHistory (empty)")
    else
        # 打印最新一步的指标
        @printf(io, "Loss: %10.6f  Accuracy: %10.6f (Steps: %d)", 
                h.loss[end], h.accuracy[end], length(h.loss))
    end
    print("\n")
end
```

---


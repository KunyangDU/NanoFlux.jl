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
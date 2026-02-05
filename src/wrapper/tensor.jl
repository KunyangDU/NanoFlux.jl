


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



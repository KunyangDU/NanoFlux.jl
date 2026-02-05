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
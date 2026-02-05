
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

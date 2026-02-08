
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

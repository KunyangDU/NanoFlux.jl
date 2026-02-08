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
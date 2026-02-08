# src/module/unet_blocks.jl

struct ResNetBlock <: AbstractModule
    norm1::GroupNorm
    conv1::Conv
    norm2::GroupNorm
    conv2::Conv
    time_proj::Dense # 将时间向量投影到当前通道数
    shortcut::Union{Conv, Identity} # 如果输入输出通道不同，需要 1x1 卷积
end

function ResNetBlock(in_ch::Int, out_ch::Int, time_dim::Int; groups=8)
    # Shortcut 逻辑
    shortcut = (in_ch != out_ch) ? Conv(2, in_ch, out_ch, 1) : Identity()
    
    return ResNetBlock(
        GroupNorm(groups, in_ch),
        Conv(2, in_ch, out_ch, 3, stride=1, pad=1, act=silu),
        GroupNorm(groups, out_ch),
        Conv(2, out_ch, out_ch, 3, stride=1, pad=1, act=silu),
        Dense(time_dim, out_ch, silu), # 时间投影层
        shortcut
    )
end

# 初始化逻辑略 (对每个子模块调用 initialize)

function (m::ResNetBlock)(x::SpatialTensor, t_emb::AbstractArray, ps::ParamsContainer)
    # 第一层卷积
    h = m.conv1(m.norm1(x, ps.norm1), ps.conv1)
    
    # 注入时间嵌入 (Scale & Shift 或 仅 Add)
    # DDPM 论文通常直接相加: h + dense(t_emb)
    # t_emb: (TimeDim, Batch) -> proj -> (OutCh, Batch)
    t_proj = m.time_proj(FlatTensor(t_emb), ps.time_proj).data
    
    # 广播加法: (W, H, C, B) + (1, 1, C, B)
    # 需要 reshape t_proj
    t_proj_reshaped = reshape(t_proj, 1, 1, size(t_proj)...)
    h = SpatialTensor{2}(h.data .+ t_proj_reshaped)
    
    # 第二层卷积
    h = m.conv2(m.norm2(h, ps.norm2), ps.conv2)
    
    # 残差连接
    sc = (m.shortcut isa Identity) ? x : m.shortcut(x, ps.shortcut)
    
    return h + sc
end
function initialize(m::ResNetBlock, rng::TaskLocalRNG = Random.default_rng())
    return (
        # 必须显式初始化每一个子模块，并给它们起对应的名字
        norm1     = initialize(m.norm1, rng),
        conv1     = initialize(m.conv1, rng),
        norm2     = initialize(m.norm2, rng),
        conv2     = initialize(m.conv2, rng),
        time_proj = initialize(m.time_proj, rng),
        shortcut  = initialize(m.shortcut, rng)
    )
end
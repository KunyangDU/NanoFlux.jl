struct UNet <: AbstractModule
    time_embed::TimeEmbedding
    
    # 简单的两层架构示例 (32 -> 16 -> 8)
    inc::Conv
    
    # Down 1 (32 -> 16)
    down1_1::ResNetBlock
    down1_2::ResNetBlock
    down1_sample::Conv # stride=2
    
    # Down 2 (16 -> 8)
    down2_1::ResNetBlock
    down2_2::ResNetBlock
    down2_attn::SpatialAttention # 在 8x8 处加 Attention
    down2_sample::Conv # stride=2
    
    # Middle (8x8)
    mid_1::ResNetBlock
    mid_attn::SpatialAttention
    mid_2::ResNetBlock
    
    # Up 2 (8 -> 16)
    up2_0::Conv # Upsample
    up2_1::ResNetBlock # Input: 拼接后的通道
    up2_2::ResNetBlock
    
    # Up 1 (16 -> 32)
    up1_0::Conv
    up1_1::ResNetBlock
    up1_2::ResNetBlock
    
    out::Sequential # Norm -> SiLU -> Conv
end

# Forward 逻辑需要手动管理 Skip Connections
function (m::UNet)(x::SpatialTensor, t::Vector{Int}, ps::ParamsContainer)
    # 1. Time Embed
    t_emb = m.time_embed(t, ps.time_embed)
    
    # 2. Input
    h_inc = m.inc(x, ps.inc)
    
    # 3. Down 1 (32x32)
    h_d1 = m.down1_1(h_inc, t_emb, ps.down1_1)
    h_d1 = m.down1_2(h_d1, t_emb, ps.down1_2)
    # [关键] 保存 h_d1 用于后面的 skip connection，而不是 push!
    
    h_d1_sample = m.down1_sample(h_d1, ps.down1_sample) # 下采样 -> 16x16
    
    # 4. Down 2 (16x16)
    h_d2 = m.down2_1(h_d1_sample, t_emb, ps.down2_1)
    h_d2 = m.down2_2(h_d2, t_emb, ps.down2_2)
    h_d2 = m.down2_attn(h_d2, ps.down2_attn)
    # [关键] 保存 h_d2 用于后面的 skip connection
    
    h_d2_sample = m.down2_sample(h_d2, ps.down2_sample) # 下采样 -> 8x8
    
    # 5. Middle (8x8)
    h_mid = m.mid_1(h_d2_sample, t_emb, ps.mid_1)
    h_mid = m.mid_attn(h_mid, ps.mid_attn)
    h_mid = m.mid_2(h_mid, t_emb, ps.mid_2)
    
    # 6. Up 2 (8 -> 16)
    h_up2 = upsample(h_mid)
    h_up2 = m.up2_0(h_up2, ps.up2_0)
    
    # [关键] 直接使用 h_d2 进行拼接，而不是 pop!
    h_up2 = cat_channels(h_up2, h_d2)
    
    h_up2 = m.up2_1(h_up2, t_emb, ps.up2_1)
    h_up2 = m.up2_2(h_up2, t_emb, ps.up2_2)
    
    # 7. Up 1 (16 -> 32)
    h_up1 = upsample(h_up2)
    h_up1 = m.up1_0(h_up1, ps.up1_0)
    
    # [关键] 直接使用 h_d1 进行拼接
    h_up1 = cat_channels(h_up1, h_d1)
    
    h_up1 = m.up1_1(h_up1, t_emb, ps.up1_1)
    h_up1 = m.up1_2(h_up1, t_emb, ps.up1_2)
    
    # 8. Out
    return m.out(h_up1, ps.out)
end

# 放在 src/network/UNet.jl 中

function initialize(m::UNet, rng::TaskLocalRNG = Random.default_rng())
    return (
        time_embed   = initialize(m.time_embed, rng),
        inc          = initialize(m.inc, rng),
        
        # Down 1
        down1_1      = initialize(m.down1_1, rng),
        down1_2      = initialize(m.down1_2, rng),
        down1_sample = initialize(m.down1_sample, rng),
        
        # Down 2
        down2_1      = initialize(m.down2_1, rng),
        down2_2      = initialize(m.down2_2, rng),
        down2_attn   = initialize(m.down2_attn, rng),
        down2_sample = initialize(m.down2_sample, rng),
        
        # Middle
        mid_1        = initialize(m.mid_1, rng),
        mid_attn     = initialize(m.mid_attn, rng),
        mid_2        = initialize(m.mid_2, rng),
        
        # Up 2
        up2_0        = initialize(m.up2_0, rng),
        up2_1        = initialize(m.up2_1, rng),
        up2_2        = initialize(m.up2_2, rng),
        
        # Up 1
        up1_0        = initialize(m.up1_0, rng),
        up1_1        = initialize(m.up1_1, rng),
        up1_2        = initialize(m.up1_2, rng),
        
        # Out
        out          = initialize(m.out, rng)
    )
end
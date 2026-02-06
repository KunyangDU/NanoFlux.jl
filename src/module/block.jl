# 建议直接叫 Block，放在 Transformer 模块下
struct Block{A, N, M} <: AbstractModule
    ln1::N    # LayerNorm
    attn::A   # Causal Attention
    ln2::N    # LayerNorm
    mlp::M    # FeedForward (MLP)
end

function Block(embed_dim::Int, heads::Int, seq_len::Int)
    # 注意：GPT 系列通常使用 GELU
    mlp_hidden = 4 * embed_dim
    return Block(
        LayerNorm(embed_dim),
        Attention(embed_dim, heads, seq_len), 
        LayerNorm(embed_dim),
        Sequential(
            # Input((embed_dim, 1)),
            Dense(embed_dim, mlp_hidden, gelu),
            Dense(mlp_hidden, embed_dim)
        )
    )
end

# GPT-2 / GPT-3 风格的 Forward (Pre-Norm)
function (m::Block)(x::AbstractNanoTensor, ps::ParamsContainer)
    # 1. 路径 A: Norm -> Attention
    # 注意：x 保持原样进入残差，而 norm 后的数据进入计算
    x_norm1 = m.ln1(x, ps.ln1)
    attn_out = m.attn(x_norm1, ps.attn) # 这里的 attn 必须是 causal 的
    
    # 2. 残差连接 1
    x = x + attn_out
    
    # 3. 路径 B: Norm -> MLP
    x_norm2 = m.ln2(x, ps.ln2)
    mlp_out = m.mlp(x_norm2, ps.mlp)
    
    # 4. 残差连接 2
    return x + mlp_out
end

function initialize(m::Block, rng::TaskLocalRNG = Random.default_rng())
    return (
        # 这里的键名 (ln1, attn...) 必须和前向传播中 ps.xxx 的访问一致
        ln1  = initialize(m.ln1, rng),
        attn = initialize(m.attn, rng),
        ln2  = initialize(m.ln2, rng),
        mlp  = initialize(m.mlp, rng)
    )
end
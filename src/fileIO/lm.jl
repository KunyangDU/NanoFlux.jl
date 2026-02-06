
struct CausalLMDataset{T, F}
    block_size::Int
    windows::T
    transform::F # 新增：转换函数
end

function CausalLMDataset(tokens::Vector{Int}, block_size::Int; transform=identity)
    data_matrix = reshape(tokens, 1, :)
    sw = slidingwindow(data_matrix; size=block_size+1, stride=1, obsdim=2)
    return CausalLMDataset(block_size, sw, transform)
end

# --- MLUtils 接口适配 ---

MLUtils.numobs(d::CausalLMDataset) = length(d.windows)

function MLUtils.getobs(d::CausalLMDataset, i::Int)
    w = vec(d.windows[i])
    x, y = w[1:end-1], w[2:end]
    # 通过钩子允许外部注入数据增强逻辑（如 Dropout 或随机扰动）
    return d.transform((x, y))
end

function MLUtils.getobs(d::CausalLMDataset, indices::AbstractVector{<:Integer})
    samples = [MLUtils.getobs(d, i) for i in indices]
    
    batch_x = reduce(hcat, first.(samples))
    batch_y = reduce(hcat, last.(samples))
    
    return (batch_x, batch_y)
end

function Base.show(io::IO, d::CausalLMDataset)
    print(io, "CausalLMDataset(Window=$(d.block_size), Obs=$(numobs(d)))")
end
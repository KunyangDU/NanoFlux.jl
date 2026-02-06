
struct Dense{F} <: AbstractModule
    in_dim::Int
    out_dim::Int
    act::F
end
Dense(in::Int, out::Int, act=identity) = Dense{typeof(act)}(in, out, act)

(l::Dense)(x::FlatTensor, ps::ParamsContainer) = FlatTensor(l.act.(ps.W * x.data .+ ps.b))

function (l::Dense)(x::SpatialTensor{D}, ps::ParamsContainer) where D
    # x.data 形状: (Features, Spatial..., Batch)
    # 对于 GPT: (Embed, Seq, Batch) -> D=1
    
    raw_data = x.data
    sz = size(raw_data)
    in_features = sz[1]
    
    # 检查维度匹配
    if in_features != l.in_dim
        error("Dense Layer Mismatch! Expected input dim $(l.in_dim), got $in_features")
    end

    # 策略：这是 Pointwise 操作 (对每个点独立做 Dense)
    # 我们把 (Features, A, B, C...) 视为 (Features, Total_Points)
    # 这样可以用一次矩阵乘法完成所有计算，效率最高
    
    flat_input = reshape(raw_data, in_features, :) # (In, N)
    
    # (Out, In) * (In, N) -> (Out, N)
    flat_out = ps.W * flat_input .+ ps.b
    flat_act = l.act.(flat_out)
    
    # 把第一维从 In 变成 Out，后面的维度保持原样
    new_size = (l.out_dim, sz[2:end]...)
    final_data = reshape(flat_act, new_size)
    
    return SpatialTensor{D}(final_data)
end

function Base.show(io::IO, l::Dense)
    print(io, "Dense($(l.in_dim) => $(l.out_dim))")
    l.act != identity && print(io, ", $(l.act)")
end
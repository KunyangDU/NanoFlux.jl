
struct Conv{D, F} <: AbstractModule
    in_ch::Int   # 需要记录这些以进行初始化
    out_ch::Int
    k_size::NTuple{D, Int}
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    act::F
end

function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; stride=1, dilation=1, act=identity)
    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    return Conv{D, typeof(act)}(in_ch, out_ch, ks, st, di, act)
end

for N in 1:3    
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N}, ps::ParamsContainer)
            y = NNlib.conv(x.data, ps.W; stride=l.stride, dilation=l.dilation, pad=0)
            
            # 处理 Bias
            # y 的形状是 (W_out, H_out, C_out, B)
            # ps.b 的形状是 (C_out,)
            # 我们需要将 b reshape 为 (1, 1, C_out, 1) 才能正确广播
            
            # 生成 reshape 维度: 前面 N 个 1，中间是 C，最后是 1
            # 例如 2D 卷积: (1, 1, C_out, 1)
            bias_shape = (ntuple(_->1, $N)..., length(ps.b), 1)
            
            return SpatialTensor{$N}(l.act.(y .+ reshape(ps.b, bias_shape)))
        end
    end
end


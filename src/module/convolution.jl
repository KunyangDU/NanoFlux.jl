
struct Conv{D, F, P} <: AbstractModule
    in_ch::Int   # 需要记录这些以进行初始化
    out_ch::Int
    k_size::NTuple{D, Int}
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    pad::P  # 新增: Padding 参数
    act::F
end

function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; 
              stride=1, dilation=1, pad=0, act=identity)

    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    pd = pad isa Int ? ntuple(_->pad, D) : pad
    
    return Conv{D, typeof(act), typeof(pd)}(in_ch, out_ch, ks, st, di, pd, act)
end
for N in 1:3    
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N}, ps::ParamsContainer)
            y = NNlib.conv(x.data, ps.W; stride=l.stride, dilation=l.dilation, pad=l.pad)
            bias_shape = (ntuple(_->1, $N)..., length(ps.b), 1)
            return SpatialTensor{$N}(l.act.(y .+ reshape(ps.b, bias_shape)))
        end
    end
end

function initialize(l::Conv{D}, rng::TaskLocalRNG = Random.default_rng()) where {D}
    w_shape = (l.k_size..., l.in_ch, l.out_ch) 
    fan_in = l.in_ch * prod(l.k_size)
    scale = sqrt(2.0 / fan_in)
    
    return (
        W = randn(rng, Float32, w_shape...) .* Float32(scale),
        b = zeros(Float32, l.out_ch)
    )
end

function Base.show(io::IO, l::Conv)
    print(io, "Conv($(l.in_ch) => $(l.out_ch), k=$(l.k_size)")
    all(x->x==1, l.stride) || print(io, ", s=$(l.stride)")
    all(x->x==0, l.pad)    || print(io, ", p=$(l.pad)") # 打印 padding
    all(x->x==1, l.dilation) || print(io, ", d=$(l.dilation)")
    l.act != identity && print(io, ", $(l.act)")
    print(io, ")")
end


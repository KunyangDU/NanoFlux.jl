
struct Conv{D, TW, TB, F} <: AbstractModule
    W::TW
    b::TB
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    act::F
end

function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; stride=1, dilation=1, act=identity)

    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    
    w_shape = (out_ch, in_ch, ks...)

    fan_in = in_ch * prod(ks)
    scale = sqrt(2.0 / fan_in)
    W = randn(Float32, w_shape...) .* Float32(scale)
    b = zeros(Float32, out_ch)
    
    return Conv{D, typeof(W), typeof(b), typeof(act)}(W, b, st, di, act)
end

for N in 1:3
    input_idxs = [Symbol("x_$d") for d in 1:N] # 输出坐标: x_1, x_2
    kern_idxs  = [Symbol("k_$d") for d in 1:N] # 核坐标: k_1, k_2

    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("d_$d")) = l.dilation[$d]))
        push!(unpack_exprs, :($(Symbol("ksize_$d")) = size(l.W, $d + 2)))
    end
    
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        d_sym = Symbol("d_$d")

        :(( $(input_idxs[d]) - 1 ) * $(Expr(:$, s_sym)) + ( $(kern_idxs[d]) - 1 ) * $(Expr(:$, d_sym)) + 1)
    end

    # H_out = (H_in - dilation*(k-1) - 1) ÷ stride + 1
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - ($(Symbol("d_$d")) * ($(Symbol("ksize_$d")) - 1) + 1)) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]

    # (x_1 in 1:out_dim_1, x_2 in 1:out_dim_2)
    ranges = Expr(:tuple, [ :($(input_idxs[d]) in 1:$(Symbol("out_dim_$d"))) for d in 1:N ]...)

    left_side = :(y[o, $(input_idxs...), b])
    
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N})
            X = x.data
            W = l.W
            b_vec = l.b
            in_size = size(X)
            
            $(unpack_exprs...)
            $(size_calc_exprs...)
            @tullio $left_side := W[o, c, $(kern_idxs...)] * X[c, $(access_exprs...), b] $ranges
            
            bias_shape = (length(b_vec), $(ones(Int, N)...), 1)
            y = y .+ reshape(b_vec, bias_shape)
            
            return SpatialTensor{$N}(l.act.(y))
        end
    end
end


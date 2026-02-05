# D: 空间维度
# TW: 权重类型 (Tensor Weight) -> 4D/5D Array
# TB: 偏置类型 (Tensor Bias)   -> 1D Vector
# F: 激活函数
struct Conv{D, TW, TB, F} <: AbstractModule
    W::TW
    b::TB
    stride::NTuple{D, Int}
    dilation::NTuple{D, Int}
    act::F
end
# 智能构造函数
function Conv(D::Int, in_ch::Int, out_ch::Int, k_size::Union{Int, NTuple}; stride=1, dilation=1, act=identity)
    
    # 1. 规范化参数为 NTuple
    # 如果用户只传了 k=3，自动变成 (3, 3, ...)
    ks = k_size isa Int ? ntuple(_->k_size, D) : k_size
    st = stride isa Int ? ntuple(_->stride, D) : stride
    di = dilation isa Int ? ntuple(_->dilation, D) : dilation
    
    # 2. 初始化权重
    # 权重形状: (Out, In, k_1, k_2, ..., k_D)
    w_shape = (out_ch, in_ch, ks...)
    
    # He Initialization
    fan_in = in_ch * prod(ks)
    scale = sqrt(2.0 / fan_in)
    W = randn(Float32, w_shape...) .* Float32(scale)
    b = zeros(Float32, out_ch)
    
    return Conv{D, typeof(W), typeof(b), typeof(act)}(W, b, st, di, act)
end

for N in 1:3
    # A. 准备符号
    input_idxs = [Symbol("x_$d") for d in 1:N] # 输出坐标: x_1, x_2
    kern_idxs  = [Symbol("k_$d") for d in 1:N] # 核坐标: k_1, k_2
    
    # B. 准备参数解包代码 (生成 s_1 = l.stride[1] 等)
    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("d_$d")) = l.dilation[$d]))
        push!(unpack_exprs, :($(Symbol("ksize_$d")) = size(l.W, $d + 2)))
    end
    
    # C. 构造索引公式 (关键修复！)
    # 我们需要生成的代码类似于: (x_1 - 1) * $s_1 + ...
    # 在元编程中，要生成 '$s_1'，需要写成 Expr(:$, :s_1)
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        d_sym = Symbol("d_$d")
        
        # 构造: (x - 1) * $s + (k - 1) * $d + 1
        # 注意这里的 Expr(:$, s_sym) 是为了让 Tullio 看到 $s_1
        :(( $(input_idxs[d]) - 1 ) * $(Expr(:$, s_sym)) + ( $(kern_idxs[d]) - 1 ) * $(Expr(:$, d_sym)) + 1)
    end
    
    # D. 构造输出尺寸计算代码
    # H_out = (H_in - dilation*(k-1) - 1) ÷ stride + 1
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - ($(Symbol("d_$d")) * ($(Symbol("ksize_$d")) - 1) + 1)) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]
    
    # E. 构造循环范围 Tuple
    # (x_1 in 1:out_dim_1, x_2 in 1:out_dim_2)
    ranges = Expr(:tuple, [ :($(input_idxs[d]) in 1:$(Symbol("out_dim_$d"))) for d in 1:N ]...)

    # F. 构造 Tullio 左值
    left_side = :(y[o, $(input_idxs...), b])
    
    # --- 生成最终函数 ---
    @eval begin
        function (l::Conv{$N})(x::SpatialTensor{$N})
            X = x.data
            W = l.W
            b_vec = l.b
            
            # 1. 获取输入尺寸 (注意: index 1 是 channel，所以空间维从 2 开始)
            in_size = size(X)
            
            # 2. 解包 stride, dilation (s_1, d_1...)
            $(unpack_exprs...)
            
            # 3. 计算输出尺寸 (out_dim_1...)
            $(size_calc_exprs...)
            
            # 4. 核心计算 (已插入 $ 符号)
            @tullio $left_side := W[o, c, $(kern_idxs...)] * X[c, $(access_exprs...), b] $ranges
            
            # 5. Bias 广播
            # 形状: (Out, 1..., 1)
            bias_shape = (length(b_vec), $(ones(Int, N)...), 1)
            y = y .+ reshape(b_vec, bias_shape)
            
            return SpatialTensor{$N}(l.act.(y))
        end
    end
end


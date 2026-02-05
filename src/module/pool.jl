
# ============================
# 1. 结构体定义
# ============================
struct Pool{D, F} <: AbstractModule
    k::NTuple{D, Int}
    stride::NTuple{D, Int}
    mode::F # mean 或 maximum
end

# 智能构造函数
function Pool(D::Int, k::Union{Int, NTuple}; 
              stride=k, mode=mean) # 默认 stride=k (不重叠)
    
    ks = k isa Int ? ntuple(_->k, D) : k
    st = stride isa Int ? ntuple(_->stride, D) : stride
    
    return Pool{D, typeof(mode)}(ks, st, mode)
end

for N in 1:3
    # A. 准备符号
    out_idx = [Symbol("x_$d") for d in 1:N] # 输出索引 x_1
    red_idx = [Symbol("i_$d") for d in 1:N] # 归约索引 i_1
    
    # B. 准备解包代码 (生成 s_1 = l.stride[1] ...)
    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("k_$d")) = l.k[$d]))
    end
    
    # C. 手动构建坐标公式 (避免 $ 语法错误)
    # 目标公式: (x_1 - 1) * $s_1 + i_1
    # 我们用 Expr(:call, ...) 手动拼装，而不是写在 :(...) 里
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        
        # 1. 构造 "$s_1" 这个节点
        s_node = Expr(:$, s_sym) 
        
        # 2. 构造 (x_1 - 1)
        sub_node = :($(out_idx[d]) - 1)
        
        # 3. 构造 (x - 1) * $s
        mult_node = Expr(:call, :*, sub_node, s_node)
        
        # 4. 构造 ... + i_1
        final_node = Expr(:call, :+, mult_node, red_idx[d])
        
        return final_node
    end
    
    # D. 构造输出尺寸计算代码
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - $(Symbol("k_$d"))) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]
    
    # E. 构造循环范围
    # 注意: Tullio 的范围不需要 $ (如 i in 1:k_1 即可)，只需要公式里有 $
    ranges_list = []
    for d in 1:N
        # 输出范围: x_1 in 1:out_dim_1
        push!(ranges_list, :($(out_idx[d]) in 1:$(Symbol("out_dim_$d"))))
        # 归约范围: i_1 in 1:k_1
        push!(ranges_list, :($(red_idx[d]) in 1:$(Symbol("k_$d"))))
    end
    ranges_tuple = Expr(:tuple, ranges_list...) # 拼成 (r1, r2...)

    # F. 生成函数
    @eval begin
        # Mean Pooling
        function (l::Pool{$N})(::typeof(mean), x::SpatialTensor{$N})
            X = x.data
            in_size = size(X)
            $(unpack_exprs...)
            $(size_calc_exprs...)
            
            # 手动拼接 @tullio 调用
            # 注意：access_exprs 里面已经包含了我们手动构造的 $ 节点
            # @tullio y[c, $(out_idx...), b] := mean(X[c, $(access_exprs...), b]) $ranges_tuple
            @tullio y[c, $(out_idx...), b] := mean(X[c, $(access_exprs...), b]) $ranges_tuple grad=Dual

            return SpatialTensor{$N}(y)
        end

        # Max Pooling
        function (l::Pool{$N})(::typeof(maximum), x::SpatialTensor{$N})
            X = x.data
            in_size = size(X)
            $(unpack_exprs...)
            $(size_calc_exprs...)
            
            # @tullio y[c, $(out_idx...), b] := maximum(X[c, $(access_exprs...), b]) $ranges_tuple
            @tullio y[c, $(out_idx...), b] := maximum(X[c, $(access_exprs...), b]) $ranges_tuple grad=Dual
            
            return SpatialTensor{$N}(y)
        end
    end
end

# ============================
# 3. 统一入口
# ============================
function (l::Pool)(x::SpatialTensor)
    return l(l.mode, x)
end

# _params(::Params, ::Pooling) = nothing


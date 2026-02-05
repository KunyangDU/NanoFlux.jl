

struct Pool{D, F} <: AbstractModule
    k::NTuple{D, Int}
    stride::NTuple{D, Int}
    mode::F # mean 或 maximum
end

function Pool(D::Int, k::Union{Int, NTuple}; 
              stride=k, mode=mean) # 默认 stride=k (不重叠)
    
    ks = k isa Int ? ntuple(_->k, D) : k
    st = stride isa Int ? ntuple(_->stride, D) : stride
    
    return Pool{D, typeof(mode)}(ks, st, mode)
end

for N in 1:3
    out_idx = [Symbol("x_$d") for d in 1:N] # 输出索引 x_1
    red_idx = [Symbol("i_$d") for d in 1:N] # 归约索引 i_1
    
    unpack_exprs = []
    for d in 1:N
        push!(unpack_exprs, :($(Symbol("s_$d")) = l.stride[$d]))
        push!(unpack_exprs, :($(Symbol("k_$d")) = l.k[$d]))
    end
    
    access_exprs = map(1:N) do d
        s_sym = Symbol("s_$d")
        
        s_node = Expr(:$, s_sym) 
        
        sub_node = :($(out_idx[d]) - 1)
        
        mult_node = Expr(:call, :*, sub_node, s_node)
        
        final_node = Expr(:call, :+, mult_node, red_idx[d])
        
        return final_node
    end
    
    size_calc_exprs = [
        :($(Symbol("out_dim_$d")) = (in_size[$d+1] - $(Symbol("k_$d"))) ÷ $(Symbol("s_$d")) + 1)
        for d in 1:N
    ]
    
    ranges_list = []
    for d in 1:N
        push!(ranges_list, :($(out_idx[d]) in 1:$(Symbol("out_dim_$d"))))
        push!(ranges_list, :($(red_idx[d]) in 1:$(Symbol("k_$d"))))
    end
    ranges_tuple = Expr(:tuple, ranges_list...) # 拼成 (r1, r2...)

    @eval begin
        # Mean Pooling
        function (l::Pool{$N})(::typeof(mean), x::SpatialTensor{$N}, ps::ParamsContainer)
            X = x.data
            in_size = size(X)
            $(unpack_exprs...)
            $(size_calc_exprs...)
            
            # @tullio y[c, $(out_idx...), b] := mean(X[c, $(access_exprs...), b]) $ranges_tuple
            @tullio y[c, $(out_idx...), b] := mean(X[c, $(access_exprs...), b]) $ranges_tuple grad=Dual

            return SpatialTensor{$N}(y)
        end

        # Max Pooling
        function (l::Pool{$N})(::typeof(maximum), x::SpatialTensor{$N}, ps)
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

function (l::Pool)(x::SpatialTensor, ps::ParamsContainer)
    return l(l.mode, x, ps)
end



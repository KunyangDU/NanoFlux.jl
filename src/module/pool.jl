

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
    @eval begin
        # Mean Pooling
        function (l::Pool{$N})(::typeof(mean), x::SpatialTensor{$N}, ps::ParamsContainer)
            y = NNlib.meanpool(x.data, l.k; stride=l.stride)
            return SpatialTensor{$N}(y)
        end

        # Max Pooling
        function (l::Pool{$N})(::typeof(maximum), x::SpatialTensor{$N}, ps)
            y = NNlib.maxpool(x.data, l.k; stride=l.stride)
            return SpatialTensor{$N}(y)
        end
    end
end

function (l::Pool)(x::SpatialTensor, ps::ParamsContainer)
    return l(l.mode, x, ps)
end



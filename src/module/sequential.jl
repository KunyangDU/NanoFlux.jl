struct Sequential{T} <: AbstractModule
    layers::T
    function Sequential(layers...)
        S = new{typeof(layers)}(layers)
        _check(S)
        return S
    end
end

Base.iterate(S::Sequential) = iterate(S.layers)
Base.iterate(S::Sequential, state) = iterate(S.layers, state)
Base.length(S::Sequential) = length(S.layers)
Base.getindex(S::Sequential, i) = getindex(S.layers, i)
Base.lastindex(S::Sequential) = lastindex(S.layers)
Base.eltype(::Type{Sequential}) = AbstractModule


# function (model::Sequential)(x, ps::ParamsContainer)
#     for (i, layer) in enumerate(model.layers)
#         x = layer(x, ps[i])
#     end
#     return x
# end

@inline function chain(x, layers::Tuple, ps::Union{Tuple, NamedTuple})
    layer = layers[1]
    p = ps[1]
    out = layer(x, p)
    return chain(out, Base.tail(layers), Base.tail(ps))
end

@inline chain(x, ::Tuple{}, ::Union{Tuple, NamedTuple}) = x

(model::Sequential)(x, ps::Union{Tuple, NamedTuple}) = chain(x, model.layers, ps)

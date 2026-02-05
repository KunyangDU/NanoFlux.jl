struct Sequential <: AbstractModule
    layers::Vector{AbstractModule}
    function Sequential(A::AbstractVector)
        S = new(convert(Vector{AbstractModule},A))
        _check(S)
        return S
    end
end
Sequential(layers...) = Sequential(collect(layers))

Base.iterate(S::Sequential) = iterate(S.layers)
Base.iterate(S::Sequential, state) = iterate(S.layers, state)
Base.length(S::Sequential) = length(S.layers)
Base.getindex(S::Sequential, i) = getindex(S.layers, i)
Base.lastindex(S::Sequential) = lastindex(S.layers)
Base.eltype(::Type{Sequential}) = AbstractModule

function (model::Sequential)(x)
    for layer in model.layers
        x = layer(x)
    end
    return x
end

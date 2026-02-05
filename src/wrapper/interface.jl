
Base.size(t::FlatTensor) = size(t.data)
Base.size(t::FlatTensor, d::Int) = size(t.data, d)
Base.length(t::FlatTensor) = length(t.data)
Base.getindex(t::FlatTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::FlatTensor, v, i...) = setindex!(t.data, v, i...)
Base.IndexStyle(::Type{<:FlatTensor}) = IndexLinear()
Base.eltype(::FlatTensor{T}) where T = T


function Base.show(io::IO, t::FlatTensor)
    dims = size(t)
    print(io, "FlatTensor($(eltype(t)), $(join(dims, "×")))")
end

function Base.show(io::IO, ::MIME"text/plain", t::FlatTensor)
    summary(io, t)
    println(io, ":")
    Base.print_array(io, t.data)
end

Base.summary(io::IO, t::FlatTensor) = print(io, "FlatTensor($(eltype(t)), $(join(size(t), "×")))")

function Base.similar(t::FlatTensor, ::Type{T}, dims::Dims) where {T}
    if length(dims) == 2
        return FlatTensor(similar(t.data, T, dims))
    else
        return similar(t.data, T, dims)
    end
end


Base.size(t::AbstractNanoTensor) = size(t.data)
Base.size(t::AbstractNanoTensor, d::Int) = size(t.data, d)
Base.length(t::AbstractNanoTensor) = length(t.data)

Base.getindex(t::AbstractNanoTensor, i...) = getindex(t.data, i...)
Base.setindex!(t::AbstractNanoTensor, v, i...) = setindex!(t.data, v, i...)

Base.IndexStyle(::Type{<:AbstractNanoTensor}) = IndexLinear()

Base.eltype(::AbstractNanoTensor{T}) where T = T

function Base.show(io::IO, t::SpatialTensor{D}) where D
    dims = size(t)
    print(io, "SpatialTensor{$D}($(eltype(t)), $(join(dims, "×")))")
end

function Base.show(io::IO, t::FlatTensor)
    dims = size(t)
    print(io, "FlatTensor($(eltype(t)), $(join(dims, "×")))")
end

function Base.show(io::IO, ::MIME"text/plain", t::AbstractNanoTensor)
    print(io, typeof(t), " with size ", size(t), ":\n")
    Base.print_array(io, t.data)
end

Base.similar(t::SpatialTensor{D}, ::Type{T}, dims::Dims) where {D, T} = 
    SpatialTensor{D}(similar(t.data, T, dims))

Base.similar(t::FlatTensor, ::Type{T}, dims::Dims) where {T} = 
    FlatTensor(similar(t.data, T, dims))
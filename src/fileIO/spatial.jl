
struct SpatialDataset{D, X, Y}
    features::X
    targets::Y
    num_classes::Int
end

function SpatialDataset(x::AbstractArray, y::AbstractVector; num_classes=0, add_channel_dim=true)
    raw_dims = ndims(x)
    n_samples = size(x)[end]
    
    D = add_channel_dim ? raw_dims - 1 : raw_dims - 2
    
    x_proc = add_channel_dim ? reshape(x, size(x)[1:end-1]..., 1, n_samples) : x
    x_proc = Float32.(x_proc)
    
    y_proc = num_classes > 0 ? Float32.(onehotbatch(y, 0:(num_classes-1))) : y
    
    return SpatialDataset{D, typeof(x_proc), typeof(y_proc)}(x_proc, y_proc, num_classes)
end

MLUtils.numobs(d::SpatialDataset) = size(d.features)[end]

function MLUtils.getobs(d::SpatialDataset, i)
    idx = (ntuple(_ -> :, ndims(d.features)-1)..., i)
    return (d.features[idx...], d.targets[:, i])
end

function Base.show(io::IO, d::SpatialDataset{D}) where D
    print(io, "SpatialDataset{$(D)D}(Samples=$(numobs(d)), Classes=$(d.num_classes))")
end
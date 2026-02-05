
struct Dense{F} <: AbstractModule
    in_dim::Int
    out_dim::Int
    act::F
end
Dense(in::Int, out::Int, act=identity) = Dense{typeof(act)}(in, out, act)

(l::Dense)(x::FlatTensor, ps::ParamsContainer) = FlatTensor(l.act.(ps.W * x.data .+ ps.b))

function (l::Dense)(x::SpatialTensor, ps::ParamsContainer)
    raw_data = x.data
    full_size = size(raw_data)
    batch_size = full_size[end]
    
    flat_features = length(raw_data) ÷ batch_size
    
    expected_in = size(ps.W, 2)
    
    if flat_features != expected_in
        error(b"❌ Auto-Flatten Dimension Mismatch!\n" *
              "Dense layer expects input dim: $(expected_in)\n" *
              "But incoming tensor $(full_size) flattens to: $(flat_features)\n" *
              "Check your Conv/Pool parameters.")
    end

    flat_data = reshape(raw_data, flat_features, batch_size)

    return l(FlatTensor(flat_data), ps::ParamsContainer)
end
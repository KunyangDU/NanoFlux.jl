
struct Dense{TW, TB, F} <: AbstractModule
    W::TW  # (Out_Dim, In_Dim)
    b::TB  # (Out_Dim)
    act::F
end

function Dense(in_dim::Int, out_dim::Int, act=identity)

    scale = sqrt(2.0f0 / in_dim)
    W = randn(Float32, out_dim, in_dim) .* scale
    

    b = zeros(Float32, out_dim)
    
    return Dense{typeof(W), typeof(b), typeof(act)}(W, b, act)
end


function (l::Dense)(x::FlatTensor)

    y_linear = l.W * x.data
    y_pre_act = y_linear .+ l.b
    
    y = l.act.(y_pre_act)

    return FlatTensor(y)
end

function (l::Dense)(x::SpatialTensor)
    raw_data = x.data
    full_size = size(raw_data)
    batch_size = full_size[end]
    
    flat_features = length(raw_data) ÷ batch_size
    
    expected_in = size(l.W, 2)
    
    if flat_features != expected_in
        error(b"❌ Auto-Flatten Dimension Mismatch!\n" *
              "Dense layer expects input dim: $(expected_in)\n" *
              "But incoming tensor $(full_size) flattens to: $(flat_features)\n" *
              "Check your Conv/Pool parameters.")
    end

    flat_data = reshape(raw_data, flat_features, batch_size)

    return l(FlatTensor(flat_data))
end
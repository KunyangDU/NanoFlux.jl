
function initialize(model::Sequential, rng::TaskLocalRNG = Random.default_rng())
    params_tuple = ntuple(i -> initialize(model.layers[i], rng), length(model.layers))
    return params_tuple
end

function initialize(l::Conv{D}, rng::TaskLocalRNG = Random.default_rng()) where D
    w_shape = (l.out_ch, l.in_ch, l.k_size...)
    fan_in = l.in_ch * prod(l.k_size)
    scale = sqrt(2.0 / fan_in)
    return (
        W = randn(rng, Float32, w_shape...) .* Float32(scale),
        b = zeros(Float32, l.out_ch)
    )
end

function initialize(l::Dense, rng::TaskLocalRNG = Random.default_rng())
    scale = sqrt(2.0f0 / l.in_dim)
    return (
        W = randn(rng, Float32, l.out_dim, l.in_dim) .* scale,
        b = zeros(Float32, l.out_dim)
    )
end

initialize(::AbstractModule,rng::TaskLocalRNG) = NamedTuple()


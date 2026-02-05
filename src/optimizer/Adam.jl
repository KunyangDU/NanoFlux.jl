
@kwdef struct Adam <: AbstractOptimizer
    learning_rate::Float32    = 1e-3
    beta1::Float32 = 0.9
    beta2::Float32 = 0.999
    eps::Float32   = 1e-8
end

mutable struct AdamState{T} <: AbstractState
    m::T
    v::T
    t::Int 
end

function initialize(opt::Adam, ps::Union{NamedTuple, Tuple})
    return map(p -> initialize(opt, p), ps)
end

function initialize(::Adam, p::AbstractArray)
    return AdamState(
        zeros(eltype(p), size(p)), # m
        zeros(eltype(p), size(p)), # v
        1                          # t (初始步数)
    )
end

initialize(::Adam, ::Any) = nothing


function update!(p::AbstractArray, g::AbstractArray, state::AdamState, opt::Adam)
    m, v = state.m, state.v
    β1, β2 = opt.beta1, opt.beta2
    ϵ = opt.eps
    

    @. m = β1 * m + (1 - β1) * g
    @. v = β2 * v + (1 - β2) * (g ^ 2)
    
    correction1 = 1 - β1 ^ state.t
    correction2 = 1 - β2 ^ state.t
    step_size = opt.learning_rate * sqrt(correction2) / correction1
    
    @. p -= step_size * m / (sqrt(v) + ϵ)
end


@kwdef struct SGD <: AbstractOptimizer
    learning_rate::Float32       = 1e-2  # 学习率
    momentum::Float32 = 0.9   # 动量系数
end
mutable struct SGDState{T<:AbstractArray} <: AbstractState
    v::T
    t::Int
end

NoOptimizer(lr::Number) = SGD(lr = convert(Float32,lr), momentum = 0.0)

function initialize(opt::SGD, ps::Union{NamedTuple, Tuple})
    return map(p -> initialize(opt, p), ps)
end

function initialize(::SGD, p::AbstractArray)
    return SGDState(
        zeros(eltype(p), size(p)), # v 初始化为全 0
        1                          # t 初始化为 1
    )
end

initialize(::SGD, ::Any) = nothing

function update!(p::AbstractArray, g::AbstractArray, state::SGDState, opt::SGD)
    @. state.v = opt.momentum * state.v + g
    @. p = p - opt.learning_rate * state.v
end
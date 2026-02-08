struct NoAlgorithm <: AbstractAlgorithm end
"""
    SimpleAlgorithm(args...)

配置训练的超参数。
支持关键字构建，例如: `SimpleAlgorithm(learning_rate=1e-3, epochs=20)`
"""
@kwdef struct TrainerConfig <: AbstractAlgorithm
    epochs::Int                          = 10
    batch_size::Int                      = 32
    show_times::Int                      = 1
    target_loss::Union{Float32, Nothing} = nothing 
    target_acc::Union{Float32, Nothing}  = nothing
    patience::Int64                      = 1
    cut_step::Number                     = Inf
    config::AbstractAlgorithm            = NoAlgorithm()
end



struct DiffusionProcess <: AbstractAlgorithm
    timesteps::Int
    beta::Vector{Float32}
    alpha::Vector{Float32}
    alpha_bar::Vector{Float32}
    sqrt_alpha_bar::Vector{Float32}
    sqrt_one_minus_alpha_bar::Vector{Float32}
end

function DiffusionProcess(timesteps::Int=1000; beta_start::Float64=1e-4, beta_end::Float64=0.02)
    # 线性调度
    beta = range(beta_start, beta_end, length=timesteps) |> collect .|> Float32
    alpha = 1.0f0 .- beta
    alpha_bar = cumprod(alpha)
    
    return DiffusionProcess(
        timesteps,
        beta,
        alpha,
        alpha_bar,
        sqrt.(alpha_bar),
        sqrt.(1.0f0 .- alpha_bar)
    )
end
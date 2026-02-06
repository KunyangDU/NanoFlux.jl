
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
end
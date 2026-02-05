# struct SimpleAlgorithm <: AbstractAlgorithm
#     batch_size::Int64
#     epochs::Int64
#     learning_rate::Float64
#     momentum::Float64
#     show_times::Number
# end
"""
    SimpleAlgorithm(args...)

配置训练的超参数。
支持关键字构建，例如: `SimpleAlgorithm(learning_rate=1e-3, epochs=20)`
"""
@kwdef struct SimpleAlgorithm <: AbstractAlgorithm
    learning_rate::Float32 = 1e-2
    momentum::Float32      = 0.9f0
    epochs::Int            = 10
    batch_size::Int        = 32
    show_times::Int        = 1
end
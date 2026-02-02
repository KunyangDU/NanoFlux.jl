struct SimpleAlgorithm <: AbstractAlgorithm
    batch_size::Int64
    epochs::Int64
    learning_rate::Float64
    momentum::Float64
    show_times::Number
end


@kwdef mutable struct TrainingHistory <: AbstractInformation
    loss::Vector{Float64}        = Float64[]
    accuracy::Vector{Float64}    = Float64[]
    count::Int64         = 1
    count_loss::Int64    = 1
    count_acc::Int64     = 1
    avg_loss::Float64    = Inf
    avg_acc::Float64     = 0.0
    to::TimerOutput      = TO
end

function Base.show(io::IO, h::TrainingHistory)
    if isempty(h.loss)
        print(io, "TrainingHistory (empty)")
    else
        @printf(io, "Loss: %10.6f  Accuracy: %10.6f  AvgLoss: %10.6f  AvgAcc: %10.6f (Steps: %d)", 
                h.loss[end], h.accuracy[end], h.avg_loss, h.avg_acc, length(h.loss))
    end
    print("\n")
end

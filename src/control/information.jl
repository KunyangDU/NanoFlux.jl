mutable struct SimpleInformation <: AbstractInformation
    loss::Vector{Float64}
    accuracy::Vector{Float64}
    velocities::Union{Nothing, IdDict}
    SimpleInformation() = new(Float64[], Float64[], nothing)
end


function Base.show(io::IO,info::SimpleInformation)
    @assert length(info.loss) == length(info.accuracy)
    _mean(x) = sum(x)/length(x)
    @printf(io,"Loss: %10.8f  Accuracy: %10.8f\n", info.loss[end],info.accuracy[end])
end

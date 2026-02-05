
# 1. 改名为 TrainingHistory，只存指标
mutable struct TrainingHistory <: AbstractInformation
    loss::Vector{Float64}
    accuracy::Vector{Float64}
    # velocities 被移除了，因为它属于训练过程中的临时状态
    
    TrainingHistory() = new(Float64[], Float64[])
end

# 2. 更安全的 show 函数
function Base.show(io::IO, h::TrainingHistory)
    if isempty(h.loss)
        print(io, "TrainingHistory (empty)")
    else
        # 打印最新一步的指标
        @printf(io, "Loss: %10.6f  Accuracy: %10.6f (Steps: %d)", 
                h.loss[end], h.accuracy[end], length(h.loss))
    end
    print("\n")
end
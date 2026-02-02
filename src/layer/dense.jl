mutable struct Dense <:  AbstractLayer
    W::AbstractTensorMap
    b::AbstractTensorMap
    activate::Function
    function Dense(Nin::Int64,Nout::Int64,act::Function = x -> max(0.0,x))
        W = TensorMap(randn, Float64, ℝ^Nout ← ℝ^Nin) / sqrt(Nin)
        b = TensorMap(zeros, Float64, ℝ^Nout ← ℝ^1)
        return new(W, b, act)
    end
end


_activate(t::AbstractTensorMap, act::Function) = TensorMap(act.(t.data), codomain(t) ← domain(t))

function _ones_tensor(x::AbstractTensorMap)
    V_batch = domain(x)
    return TensorMap(ones(eltype(x.data), 1, dim(V_batch)), ℝ^1 ← V_batch)
end


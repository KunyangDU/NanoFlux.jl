Zygote.@adjoint function TensorMap(d::AbstractArray, S)
    y = TensorMap(d, S)
    function TensorMap_pullback(Δ)
        # Δ 是回传回来的梯度。
        # 如果 Δ 已经是 TensorMap，取 .data
        # 如果 Δ 是 Array，直接用
        Δ_data = Δ isa TensorMap ? Δ.data : Δ
        return (Δ_data, nothing)
    end
    return y, TensorMap_pullback
end

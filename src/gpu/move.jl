"""
    to_device(x)

将结构体 x 中的所有数组移动到 CURRENT_DEVICE[] 指定的设备上。
支持嵌套的 NamedTuple, Tuple 以及 NanoFlux 自定义的 Tensor 类型。
"""
to_device(x) = move_to(CURRENT_DEVICE[], x)

move_to(::AbstractDevice, x::Any) = x
move_to(::CPU, x::AbstractArray) = Array(x)

function move_to(::GPU, x::AbstractArray)
    is_gpu_array(x) && return x
    METAL_AVAILABLE && return Metal.mtl(convert(AbstractArray{Float32}, x))
    CUDA_AVAILABLE && return CUDA.cu(convert(AbstractArray{Float32}, x))
    error("NanoFlux Error: CURRENT_DEVICE is GPU, but no backend (Metal/CUDA) is detected!")
end

function is_gpu_array(x::AbstractArray)
    return (METAL_AVAILABLE && x isa Metal.MtlArray) || 
           (CUDA_AVAILABLE && x isa CUDA.CuArray)
end

move_to(d::AbstractDevice, x::ParamsContainer) = map(v -> move_to(d, v), x)
move_to(d::AbstractDevice, x::AbstractDict) = Dict(k => move_to(d, v) for (k,v) in x)

move_to(d::AbstractDevice, t::SpatialTensor{D}) where D = SpatialTensor{D}(move_to(d, t.data))
move_to(d::AbstractDevice, t::FlatTensor) = FlatTensor(move_to(d, t.data))

function move_to(d::AbstractDevice, st::AdamState)
    return AdamState(
        move_to(d, st.m),
        move_to(d, st.v),
        st.t
    )
end

function move_to(d::AbstractDevice, st::SGDState)
    return SGDState(
        move_to(d, st.v),
        st.t
    )
end
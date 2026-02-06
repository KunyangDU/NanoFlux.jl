
struct CPU <: AbstractDevice end
struct GPU <: AbstractDevice end

!isdefined(@__MODULE__, :CURRENT_DEVICE) && (const CURRENT_DEVICE = Ref{AbstractDevice}(CPU()))
!isdefined(@__MODULE__, :METAL_AVAILABLE) && (const METAL_AVAILABLE = isdefined(@__MODULE__, :Metal) && Metal.functional())
!isdefined(@__MODULE__, :CUDA_AVAILABLE) && (const CUDA_AVAILABLE = isdefined(@__MODULE__, :CUDA) && CUDA.functional())


function __init__()
    # 检测 Metal 支持
    if METAL_AVAILABLE
        CURRENT_DEVICE[] = GPU() # 默认切换到 GPU
        @info "NanoFlux: Metal GPU backend detected and enabled."
    # 检测 CUDA 支持
    elseif CUDA_AVAILABLE
        CURRENT_DEVICE[] = GPU()
        @info "NanoFlux: CUDA GPU backend detected and enabled."
    end
end

__init__()

!isdefined(@__MODULE__, :TO) ? (const TO = TimerOutput()) : reset_timer!(TO)
!isdefined(@__MODULE__, :ParamsContainer) && (const ParamsContainer = Union{NamedTuple, Tuple})

manualGC() = GC.gc()

macro bg_str(s)
    # 1;32m 表示：样式1(加粗) + 颜色32(绿色)
    # 红 绿  黄
    # 31 32 33
    return "\e[1;32m" * s * "\e[0m"
end

macro g_str(s)
    return "\e[32m" * s * "\e[0m"
end
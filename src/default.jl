
if !isdefined(@__MODULE__, :TO)
    const TO = TimerOutput()
else
    reset_timer!(TO)
end

if !isdefined(@__MODULE__, :ParamsContainer)
    const ParamsContainer = Union{NamedTuple, Tuple}
end

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
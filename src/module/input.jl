"""
    Input(shape::Tuple)

一个虚拟层，仅用于在 model_summary 中记录输入形状。
在前向传播中，它是什么都不做的直通车 (Identity)。
"""
struct Input <: AbstractModule
    shape::Tuple
end

(l::Input)(x) = x

_count_params(l::Input) = 0
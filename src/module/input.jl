"""
    Input(shape::Tuple)

一个虚拟层，仅用于在 model_summary 中记录输入形状。
在前向传播中，它是什么都不做的直通车 (Identity)。
"""
struct Input <: AbstractModule
    shape::Tuple
end

# 前向传播：直接返回 x，不做任何操作，零开销
(l::Input)(x) = x

# 辅助函数：如果是 Input 层，不算参数
_count_params(l::Input) = 0
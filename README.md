# NanoFlux.jl

> **A minimalist, first-principles deep learning framework built on TensorKit.jl and Zygote.jl.**

**NanoFlux** 是一个基于 `TensorKit.jl` 和 `Zygote.jl` 从零构建的极简深度学习框架。它不依赖黑盒算子，而是将神经网络层视为严谨的线性映射（TensorMap），通过手写几何变换（Im2Col）和显式梯度流，展示了深度学习的底层数学本质。

## Versatility
Following layers are supported:
- Dense
- Convolution
- Pooling
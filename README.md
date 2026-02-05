# NanoFlux.jl
> **从第一性原理出发基于 NNlib.jl 与 Zygote.jl 构建的纯 Julia 显式梯度深度学习框架。**
## 核心特性
* **显式梯度与无状态设计 (Explicit Gradients & Stateless)**：
    模型结构 (struct) 仅存储超参数，权重与状态 (ps) 被剥离为独立的参数树，彻底解耦计算与数据。

* **严格的维度安全 (Strict Dimensional Safety)**：
    通过类型系统（SpatialTensor vs FlatTensor）强制区分“空间数据”与“特征向量”。

* **工业级高性能算子 (Industrial-Grade Kernels)**：
    底层全面接入 NNlib，利用其高度优化的 rrule。
## 支持组件
- 核心张量：`SpatialTensor{N}`, `FlatTensor`
- 网络层：`Dense`, `Conv{N}`, `Pool{N} `(Mean/Max), `Flatten`, `Input`
- 容器：`Sequential` 
- 优化器：`Adam`, `SGD`
## 性能基准
| 模型架构 | 数据集 | 准确率 | 训练耗时 | 硬件 |
| :--- | :--- | :--- | :--- | :--- |
| **LeNet-5** | MNIST | **> 98.0%** | ~ 12s | Apple M4 (10-Core) |
# NanoFlux.jl 

> **从第一性原理出发基于[Tullio.jl](https://github.com/mcabbott/Tullio.jl)、[Zygote.jl](https://github.com/FluxML/Zygote.jl)构建的极简深度学习框架。**

## 核心特性

* **严格的维度安全 (Strict Dimensional Safety)**
    通过类型系统（`SpatialTensor` vs `FlatTensor`）强制区分“空间数据”与“特征向量”。在编译期即阻断非法的张量缩并操作，从根本上消除了维度失配（Shape Mismatch）带来的隐患。

* **透明的数学内核 (Transparent Mathematical Kernels)**
    基于爱因斯坦求和约定 (Einstein Summation) 构建底层运算。每一层网络（如卷积、池化）都是可验证、可推导的显式数学公式，而非不透明的 API 调用。
## 支持组件

* **核心容器**：`SpatialTensor{N}`, `FlatTensor`
* **网络层**：`Dense`, `Conv{N}`, `Pool{N}`
* **优化器**：`SGD`, `MomentumSGD`, `Adam` 

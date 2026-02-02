
"""
桥接函数：Array -> TensorMap
这是实现并行化的关键一步
"""
function batch_to_tensormap(x_batch::AbstractMatrix, y_batch::AbstractMatrix)
    # x_batch: (784, B)
    # y_batch: (10, B)
    
    feat_dim = size(x_batch, 1)  # 784
    class_dim = size(y_batch, 1) # 10
    current_batch_size = size(x_batch, 2) # 通常是 64，最后一个 batch 可能更小
    
    # --- 关键定义 ---
    # 定义 Batch 为 Domain (源空间)
    V_batch = ℝ^current_batch_size
    
    # 定义特征为 Codomain (目标空间)
    V_in = ℝ^feat_dim
    V_out = ℝ^class_dim
    
    # 构造 TensorMap
    # 物理意义：将 Batch 空间映射到特征空间
    # 数学操作：本质上就是包装了这个 (Features, Batch) 的矩阵
    x_tensor = TensorMap(x_batch, V_in ← V_batch)
    
    # 标签同理：将 Batch 空间映射到类别空间
    y_tensor = TensorMap(y_batch, V_out ← V_batch)
    
    return x_tensor, y_tensor
end
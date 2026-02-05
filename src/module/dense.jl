# ============================
# 1. 结构体定义
# ============================
# TW: 权重类型 (Matrix)
# TB: 偏置类型 (Vector)
# F:  激活函数类型
struct Dense{TW, TB, F} <: AbstractModule
    W::TW  # (Out_Dim, In_Dim)
    b::TB  # (Out_Dim)
    act::F
end

# ============================
# 2. 构造函数
# ============================
function Dense(in_dim::Int, out_dim::Int, act=identity)
    # 初始化权重 (He Initialization / Kaiming Init)
    # 适用于 ReLU 等非线性激活
    scale = sqrt(2.0f0 / in_dim)
    W = randn(Float32, out_dim, in_dim) .* scale
    
    # 初始化偏置 (通常全0)
    b = zeros(Float32, out_dim)
    
    return Dense{typeof(W), typeof(b), typeof(act)}(W, b, act)
end

# ============================
# 3. 前向传播 (Forward)
# ============================
# 核心约束：只接受 FlatTensor
function (l::Dense)(x::FlatTensor)
    # x.data 维度: (In_Dim, Batch)
    # l.W    维度: (Out_Dim, In_Dim)
    
    # 1. 线性变换 (Matrix Multiplication)
    # (Out, In) * (In, Batch) -> (Out, Batch)
    y_linear = l.W * x.data
    
    # 2. 加上偏置 (Bias Broadcasting)
    # Julia 的 .+ 会自动把 b (Out_Dim) 广播到 (Out_Dim, Batch)
    y_pre_act = y_linear .+ l.b
    
    # 3. 激活函数
    y = l.act.(y_pre_act)
    
    # 4. 保持类型流转：输出依然是 FlatTensor
    # 这样下一个 Dense 层可以直接接收它
    return FlatTensor(y)
end

function (l::Dense)(x::SpatialTensor)
    # 1. 获取原始数据 (C, H, W, B)
    raw_data = x.data
    full_size = size(raw_data)
    batch_size = full_size[end]
    
    # 2. 计算展平后的特征数
    # 总元素数 除以 Batch数 = 单个样本的特征数
    # 例如 (16, 4, 4, 10) -> 总共 2560 -> 除以10 -> 特征数 256
    flat_features = length(raw_data) ÷ batch_size
    
    # 3. 安全检查 (这一点非常重要！)
    # 确保展平后的维度能和 Dense 的权重矩阵 W 对得上
    expected_in = size(l.W, 2)
    
    if flat_features != expected_in
        # 友好的报错信息
        error(b"❌ Auto-Flatten Dimension Mismatch!\n" *
              "Dense layer expects input dim: $(expected_in)\n" *
              "But incoming tensor $(full_size) flattens to: $(flat_features)\n" *
              "Check your Conv/Pool parameters.")
    end
    
    # 4. 执行 Reshape (Julia 的 reshape 通常是零拷贝的，非常快)
    # 变成 (Features, Batch)
    flat_data = reshape(raw_data, flat_features, batch_size)
    
    # 5. 递归调用
    # 把它包装成 FlatTensor，扔给上面的标准路径去算
    return l(FlatTensor(flat_data))
end
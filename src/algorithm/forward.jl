function forward(A::Sequential, x::AbstractTensorMap)
    x′ = deepcopy(x)
    for a in A 
        x′ = forward(a,x′)
    end
    return x′
end

forward(L::Dense, x::AbstractTensorMap) = _activate(L.W * x + L.b * _ones_tensor(x), L.activate)

function forward(l::Convolution, x_tm::AbstractTensorMap)
    # 1. 解包数据 (Features, Batch)
    x_flat = convert(Array, x_tm) 
    batch_size = size(x_flat, 2)
    H, W = l.img_size
    
    # 2. 恢复 4D 结构: (Channels, H, W, Batch)
    # 注意：MNIST 原始数据被 reshape 成 (784, B) 时，顺序是列优先
    x_img = reshape(x_flat, l.in_ch, H, W, batch_size)
    
    # 3. Im2Col
    # col_data Shape: (fan_in, Total_Windows * Batch)
    col_data, h_out, w_out = im2col_2d(x_img, l.k, l.stride, l.padding)
    
    # 4. 封装回 TensorMap 进行核心缩并
    # 为了利用 TensorKit 的矩阵乘法: output = W * col
    total_steps = size(col_data, 2) # = h_out * w_out * batch_size
    
    V_huge = ℝ^total_steps
    V_fan_in = domain(l.W) # ℝ^(fan_in)
    
    x_col_tm = TensorMap(col_data, V_fan_in ← V_huge)
    
    # 5. 卷积 (矩阵乘法) & 偏置
    # res_tm Shape: (out_ch) ← (Total_Windows * Batch)
    res_tm = l.W * x_col_tm
    
    # 偏置广播 (复用之前的 _ones_tensor 逻辑，但针对新的 domain)
    ones_vec = TensorMap(ones(eltype(x_flat), 1, total_steps), ℝ^1 ← V_huge)
    y_conv_tm = res_tm + l.b * ones_vec
    
    # 6. 激活
    y_act_tm = _activate(y_conv_tm, l.activate)
    
    # 7. 重排数据 (这是最难的一步)
    # 目前 y_act_tm 的数据是 (Out_Ch, H_out * W_out * Batch)
    # 也就是先把第一张图的所有像素排完，再排第二张图
    # 我们需要把它变成 (Out_Ch * H_out * W_out, Batch) 以便传给下一层
    
    y_raw = convert(Array, y_act_tm) # (Out_Ch, H_out*W_out*Batch)
    
    # Reshape: (Out_Ch, H_out*W_out, Batch)
    # 注意：Julia 的 reshape 是列优先填充，这里的顺序是对的
    y_reshaped = reshape(y_raw, l.out_ch * h_out * w_out, batch_size)
    
    # 8. 最终封装
    V_out_flat = ℝ^(l.out_ch * h_out * w_out)
    V_batch_new = ℝ^batch_size
    
    return TensorMap(y_reshaped, V_out_flat ← V_batch_new)
end

function forward(l::Pooling, x_tm::AbstractTensorMap)
    # 1. 解包与还原
    x_flat = convert(Array, x_tm)
    batch_size = size(x_flat, 2)
    H, W = l.img_size
    
    x_img = reshape(x_flat, l.in_ch, H, W, batch_size)
    
    # 2. 计算输出尺寸
    h_out = div(H - l.k, l.stride) + 1
    w_out = div(W - l.k, l.stride) + 1
    
    # 3. 池化操作 (利用 Julia 的 view 和 mean)
    # 预分配输出
    y_pool = Zygote.Buffer(Array{eltype(x_flat)}(undef, l.in_ch, h_out, w_out, batch_size))
    
    for b in 1:batch_size, c in 1:l.in_ch, i in 1:h_out, j in 1:w_out
        h_start = (i-1)*l.stride + 1
        w_start = (j-1)*l.stride + 1
        # 提取窗口并求平均
        patch = view(x_img, c, h_start:h_start+l.k-1, w_start:w_start+l.k-1, b)
        y_pool[c, i, j, b] = sum(patch)/length(patch)
    end
    
    # 4. 展平并封装 (Flatten)
    # 目标形状: (C * H_out * W_out, Batch)
    y_ready = copy(y_pool) # 固化 Buffer
    y_flat = reshape(y_ready, l.in_ch * h_out * w_out, batch_size)
    
    V_out = ℝ^(size(y_flat, 1))
    V_batch = ℝ^batch_size
    
    return TensorMap(y_flat, V_out ← V_batch)
end

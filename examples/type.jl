using InteractiveUtils
# 构造假数据和参数
l = Dense(64, 10, identity)
ps = initialize(l)
x = FlatTensor(randn(Float32, 64, 32))

# 检查前向传播是否有红色字体的 Any
@code_warntype l(x, ps)

using InteractiveUtils
# 构造一个假模型和假数据
model = Sequential(
    Input(28,28,1),
    Conv(2, 1, 6, 5),
    Pool(2, 2),
    Flatten(),
    Dense(6*12*12, 10)
)
ps = initialize(model)
x = SpatialTensor{2}(randn(Float32, 28, 28, 1, 1))

# 检查类型稳定性
@code_warntype model(x, ps)
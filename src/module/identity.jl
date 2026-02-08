"""
    Identity(act=identity)

恒等映射模块，支持可选的激活函数。
- `Identity()`: 不做任何操作（直通），常用于 ResNet 的 Shortcut。
- `Identity(relu)`: 仅作为激活层使用。
"""
struct Identity{F} <: AbstractModule
    act::F
end

# 默认构造函数：act 默认为 identity 函数
Identity() = Identity(identity)

# 当 act 为 identity 时，直接返回输入，避免广播开销
(::Identity{typeof(identity)})(x::AbstractNanoTensor, ::ParamsContainer) = x

# 当 act 不为 identity 时 (例如 silu)，执行广播操作
function (m::Identity)(x::SpatialTensor{D}, ::ParamsContainer) where D
    # 保持 SpatialTensor 类型
    return SpatialTensor{D}(m.act.(x.data))
end

function (m::Identity)(x::FlatTensor, ::ParamsContainer)
    # 保持 FlatTensor 类型
    return FlatTensor(m.act.(x.data))
end

# 无论是否有激活函数，Identity 都没有可训练参数
initialize(::Identity, ::TaskLocalRNG) = NamedTuple()

function Base.show(io::IO, m::Identity)
    if m.act === identity
        print(io, "Identity()")
    else
        print(io, "Identity($(m.act))")
    end
end
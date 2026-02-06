
# 定义一个节点来存储层级信息
mutable struct InspectNode
    name::String
    type_name::String
    input_shape::String
    output_shape::String
    params_count::Int
    children::Vector{InspectNode}
    depth::Int
end

"""
    summary(model::AbstractModule, input_shape::Tuple)

对任意 NanoFlux 模型进行递归结构检查、维度推导和参数统计。
"""
function summary(model::AbstractModule, input_shape::Tuple)

    full_shape = (input_shape..., 1) 
    
    spatial_dims = length(input_shape) - 1
    if spatial_dims >= 0
        x = SpatialTensor{spatial_dims}(randn(Float32, full_shape))
    else
        x = FlatTensor(randn(Float32, full_shape))
    end

    root_node = _inspect_recursive(model, x, initialize(model, Random.default_rng()), "Model", 0)

    println("="^100)
    println("Model Inspector")
    # println("="^100)
    show(root_node)
    println("-"^100)
    total_params = _sum_params(root_node)
    println("Total Parameters: $(format_number(total_params))")
    println("="^100)
    return nothing
end

# --- 核心递归逻辑 ---

function _inspect_recursive(layer::AbstractModule, x, ps, name, depth)
    in_shape_str = _fmt_shape(size(x))

    out = try
        layer(x, ps)
    catch e
        error("Dimension Mismatch in layer [$name] ($(typeof(layer))).\nInput: $in_shape_str\nError: $e")
    end
    
    out_shape_str = _fmt_shape(size(out))

    is_container = layer isa Sequential || layer isa Block # 识别容器

    self_params = is_container ? 0 : _count_elements(ps)
    
    node = InspectNode(name, string(typeof(layer)), in_shape_str, out_shape_str, self_params, [], depth)

    if layer isa Sequential
        for (i, (sub_layer, sub_ps)) in enumerate(zip(layer.layers, ps))
            sub_node = _inspect_recursive(sub_layer, x, sub_ps, "Layer $i", depth + 1)
            push!(node.children, sub_node)
            
            x = sub_layer(x, sub_ps)
        end
    elseif layer isa Block
        # Block 结构比较特殊 (ln1, attn, ln2, mlp)
        # 这种硬编码的结构需要手动拆解
        # Block Forward: x -> ln1 -> attn -> + -> ln2 -> mlp -> +
        
        # 1. LN1
        node_ln1 = _inspect_recursive(layer.ln1, x, ps.ln1, "LN1", depth + 1)
        push!(node.children, node_ln1)
        x_norm1 = layer.ln1(x, ps.ln1)
        
        # 2. Attn
        node_attn = _inspect_recursive(layer.attn, x_norm1, ps.attn, "Attention", depth + 1)
        push!(node.children, node_attn)
        # 残差连接不改变形状，直接模拟流向
        x = x + layer.attn(x_norm1, ps.attn)
        
        # 3. LN2
        node_ln2 = _inspect_recursive(layer.ln2, x, ps.ln2, "LN2", depth + 1)
        push!(node.children, node_ln2)
        x_norm2 = layer.ln2(x, ps.ln2)
        
        # 4. MLP (通常是 Sequential)
        node_mlp = _inspect_recursive(layer.mlp, x_norm2, ps.mlp, "MLP", depth + 1)
        push!(node.children, node_mlp)
    # else
        # @warn "module summary not defined!"
    end
    
    return node
end

function _sum_params(node::InspectNode)
    c = node.params_count
    for child in node.children
        c += _sum_params(child)
    end
    return c
end

_count_elements(x::Union{NamedTuple, Tuple}) = sum(_count_elements, x; init=0)
_count_elements(x::AbstractArray) = length(x)
_count_elements(x::Any) = 0
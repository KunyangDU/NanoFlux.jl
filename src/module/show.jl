

function Base.show(root::InspectNode)
    # 1. 打印表头 (Header)
    println("-"^100)
    @printf("%-45s %-22s %-22s %-10s\n", "Layer (Type)", "Input", "Output", "Param #")
    println("-"^100)

    _print_tree_recursive(root, "", true)

end


function _print_tree_recursive(node::InspectNode, prefix::String, is_last::Bool)

    if node.depth == 0
        connector = ""
        current_prefix = ""
    else
        connector = is_last ? "└─ " : "├─ "
        current_prefix = prefix
    end

    # 简化类型名：只取 Struct 名，去掉 {T...}
    simple_type = split(node.type_name, "{")[1]
    
    display_name = isempty(node.name) ? simple_type : "$(node.name) ($(simple_type))"
    
    tree_str = current_prefix * connector * display_name
    
    if length(tree_str) > 38
        tree_str = tree_str[1:35] * "..."
    end

    param_str = node.params_count > 0 ? format_number(node.params_count) : ""

    @printf("%-45s %-22s %-22s %-10s\n", 
            tree_str, 
            node.input_shape, 
            node.output_shape, 
            param_str)

    children = node.children
    count = length(children)
    
    if node.depth == 0
        next_prefix = "" # 根节点之下直接开始
    else
        next_prefix = prefix * (is_last ? "   " : "│  ")
    end

    for (i, child) in enumerate(children)
        is_last_child = (i == count)
        
        # 递归调用
        _print_tree_recursive(child, next_prefix, is_last_child)
    end
end
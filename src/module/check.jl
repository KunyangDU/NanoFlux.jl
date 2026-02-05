using Printf
using Random

"""
    _check(model::Sequential, input_shape::Tuple)

运行一次虚拟前向传播，检查层维度匹配情况，并打印详细摘要。
input_shape: (Channel, H, W) 或 (Channel, Len) 等，不包含 Batch 维度。
"""
function _check(layers::Vector{AbstractModule},input_shape::Union{Tuple, Nothing}=nothing)
    println("="^80)
    println("Model Architecture Inspector")
    println("="^80)


    # 如果用户没传 shape，尝试从第一层读取
    if input_shape === nothing
        if layers[1] isa Input
            input_shape = layers[1].shape
            # println("ℹ️  Auto-detected input shape from layer 1: $input_shape")
            # 真正的计算从第2层开始，因为第1层是虚拟的
            # 但为了显示好看，我们依然遍历它
        else
            error("Missing input_shape! \nPlease provide it as an argument OR add an Input(shape) layer at the start of your model.")
        end
    end
    
    # 1. 构造 Dummy Input (Batch Size = 1)
    # ----------------------------------------------------
    # 根据 input_shape 的长度自动判断是 SpatialTensor{1}, {2} 还是 {3}
    spatial_dims = length(input_shape) - 1 # 减去 Channel 维
    if spatial_dims < 1
        error("Input shape must be at least (Channel, Len...), got $input_shape")
    end
    
    # 构造数据 (Channel, D1, D2..., Batch=1)
    full_shape = (input_shape..., 1)
    x_data = randn(Float32, full_shape)
    
    # 包装成 Tensor
    x = SpatialTensor{spatial_dims}(x_data)
    
    println(@sprintf("Input Signal: %s (Batch=1)", string(size(x))))
    println("-"^80)
    @printf("%-4s %-15s %-25s %-25s %-10s\n", "ID", "Layer Type", "Input Shape", "Output Shape", "Params")
    println("-"^80)

    total_params = 0
    
    # 2. 逐层运行
    # ----------------------------------------------------
    for (i, layer) in enumerate(layers)
        layer_type = string(typeof(layer))
        # 只保留 Struct 名字，去掉 {...}
        layer_name = split(layer_type, "{")[1]
        
        # 记录输入形状
        in_shape = size(x)
        
        # 尝试运行该层
        try
            # --- FORWARD PASS ---
            out = layer(x)
            # --------------------
            
            out_shape = size(out)
            
            # 计算参数量
            n_params = _count_params(layer)
            total_params += n_params
            
            # 格式化打印
            str_in  = _fmt_shape(in_shape)
            str_out = _fmt_shape(out_shape)
            
            @printf("%-4d %-15s %-25s %-25s %-10d\n", 
                    i, layer_name, str_in, str_out, n_params)
            
            # 更新 x 为下一层的输入
            x = out
            
        catch e
            println("\n" * "!"^80)
            println("Layer Dimension Mismatch Detected at Layer $i [$layer_name]!")
            println("!"^80)
            println("   Expected Input: Compatible with $(_fmt_shape(in_shape))")
            
            if layer isa Dense
                println("   Layer Config:   InputDim = $(size(layer.W, 2))")
                println("   Analysis:       The Dense layer expects $(size(layer.W, 2)) features, but received $(in_shape[1]).")
                println("                   (Did you calculate the Flatten output size correctly?)")
            elseif layer isa Conv
                println("   Analysis:       Convolution failure. Check if input spatial size is smaller than Kernel size.")
            end
            
            println("\nERROR DETAIL:")
            showerror(stdout, e)
            println()
            return # 终止检查
        end
    end
    
    println("-"^80)
    println(g"CHECK PASSED")
    println("Total Parameters: $(format_number(total_params))")
    println("="^80)
end

# ---------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------

# 格式化形状字符串 (去除 Batch=1)
function _fmt_shape(s)
    # s 是 (C, H, W, B) 或 (Features, B)
    # 我们显示时不显示 Batch，显得更干净
    dims = s[1:end-1] 
    return string(dims)
end

# 计算参数量
_count_params(l::Any) = 0
_count_params(l::Dense) = length(l.W) + length(l.b)
_count_params(l::Conv)  = length(l.W) + length(l.b)

# 格式化数字 (1,234,567)
function format_number(n::Int)
    return replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
end


_check(model::Sequential) = _check(model.layers)
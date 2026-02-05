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

    if input_shape === nothing
        if layers[1] isa Input
            input_shape = layers[1].shape
        else
            error("Missing input_shape! \nPlease provide it as an argument OR add an Input(shape) layer at the start of your model.")
        end
    end
    
    spatial_dims = length(input_shape) - 1
    if spatial_dims < 1
        error("Input shape must be at least (Channel, Len...), got $input_shape")
    end

    full_shape = (input_shape..., 1)
    x_data = randn(Float32, full_shape)
    
    x = SpatialTensor{spatial_dims}(x_data)
    
    println(@sprintf("Input Signal: %s (Batch=1)", string(size(x))))
    println("-"^80)
    @printf("%-4s %-15s %-25s %-25s %-10s\n", "ID", "Layer Type", "Input Shape", "Output Shape", "Params")
    println("-"^80)

    total_params = 0
    
    for (i, layer) in enumerate(layers)
        layer_type = string(typeof(layer))
        layer_name = split(layer_type, "{")[1]
        in_shape = size(x)
        
        try
            out = layer(x)
            
            out_shape = size(out)
            
            n_params = _count_params(layer)
            total_params += n_params
            
            str_in  = _fmt_shape(in_shape)
            str_out = _fmt_shape(out_shape)
            
            @printf("%-4d %-15s %-25s %-25s %-10d\n", 
                    i, layer_name, str_in, str_out, n_params)
            
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
            return
        end
    end
    
    println("-"^80)
    println(g"CHECK PASSED")
    println("Total Parameters: $(format_number(total_params))")
    println("="^80)
end

_check(model::Sequential) = _check(model.layers)
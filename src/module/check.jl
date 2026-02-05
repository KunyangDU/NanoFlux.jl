

"""
    _check(model, input_shape)

显式梯度架构下的模型检查器。
它会通过生成临时参数并运行一次前向传播来验证形状匹配。
"""
function _check(model::Sequential, input_shape::Union{Tuple, Nothing}=nothing)
    println("="^80)
    println("Model Architecture Inspector")
    println("="^80)

    layers = model.layers
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

    full_shape = (input_shape..., 1) # Batch=1
    x_data = randn(Float32, full_shape)

    x = SpatialTensor{spatial_dims}(x_data)
    
    rng = Random.default_rng()
    full_ps = initialize(model, rng) 

    println(@sprintf("Input Signal: %s (Batch=1)", string(size(x))))
    println("-"^80)
    @printf("%-4s %-15s %-25s %-25s %-10s\n", "ID", "Layer Type", "Input Shape", "Output Shape", "Params")
    println("-"^80)

    total_params = 0
    
    for (i, (layer, layer_ps)) in enumerate(zip(layers, full_ps))
        layer_type = string(typeof(layer))
        layer_name = split(layer_type, "{")[1]
        in_shape = size(x)
        
        try
            out = layer(x, layer_ps)
            
            out_shape = size(out)
            
            n_params = _count_elements(layer_ps)
            total_params += n_params
            
            str_in  = _fmt_shape(in_shape)
            str_out = _fmt_shape(out_shape)
            
            @printf("%-4d %-15s %-25s %-25s %-10s\n", 
                    i, layer_name, str_in, str_out, format_number(n_params))
            
            x = out
            
        catch e
            println("\n" * "!"^80)
            println("Layer Dimension Mismatch Detected at Layer $i [$layer_name]!")
            println("!"^80)
            println("   Expected Input: Compatible with $(_fmt_shape(in_shape))")
            
            
            if layer isa Dense
                if haskey(layer_ps, :W)
                    expected_dim = size(layer_ps.W, 2)
                    println("   Layer Config:   InputDim = $expected_dim")
                    println("   Analysis:       The Dense layer expects $expected_dim features, but received $(in_shape[1]).")
                    println("                   (Did you calculate the Flatten output size correctly?)")
                end
            elseif layer isa Conv
                println("   Analysis:       Convolution failure.")
                println("                   Check if input spatial size is smaller than Kernel size.")
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

_check(layers::Vector{<:AbstractModule}, input_shape::Union{Tuple, Nothing}=nothing) = _check(Sequential(layers...), input_shape)

_count_elements(x::Union{NamedTuple, Tuple}) = sum(_count_elements, x; init=0)
_count_elements(x::AbstractArray) = length(x)
_count_elements(x::Any) = 0
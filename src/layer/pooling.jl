
struct Pooling <: AbstractLayer
    k::Int # Pool size
    stride::Int
    in_ch::Int
    img_size::Tuple{Int, Int} # 输入尺寸
    
    function Pooling(k::Int, in_ch::Int, img_size::Tuple{Int, Int}; stride=k)
        new(k, stride, in_ch, img_size)
    end
end

_params(::Params, ::Pooling) = nothing


struct Convolution <: AbstractLayer
    W::AbstractTensorMap    # Out_Channels ← (In_Channels * k * k)
    b::AbstractTensorMap    # Out_Channels ← 1
    k::Int
    in_ch::Int
    out_ch::Int
    img_size::Tuple{Int, Int} # (H, W) 输入尺寸，用于 reshape
    stride::Int
    padding::Int
    activate::Function
    
    function Convolution(in_ch::Int, out_ch::Int, img_size::Tuple{Int, Int}, k::Int; 
                    stride=1, padding=0, act=x -> max(0.0,x))
        
        fan_in = in_ch * k * k
        # He Initialization for ReLU
        scale = sqrt(2.0 / fan_in)
        
        W = TensorMap(randn, Float64, ℝ^out_ch ← ℝ^fan_in) * scale
        b = TensorMap(zeros, Float64, ℝ^out_ch ← ℝ^1)
        
        new(W, b, k, in_ch, out_ch, img_size, stride, padding, act)
    end
end

# 辅助函数：计算卷积后的尺寸
_out_dims(h, w, k, stride, pad) = (div(h + 2pad - k, stride) + 1, div(w + 2pad - k, stride) + 1)
function _params(ps::Params, layer::Convolution)
    push!(ps, layer.W)
    push!(ps, layer.b)
end

"""
im2col_2d: 将图片转换为矩阵列，以便进行矩阵乘法
x: (Channels, Height, Width, Batch) 
return: (k*k*Channels, H_out*W_out*Batch)
"""
function im2col_2d(x::AbstractArray{T, 4}, k::Int, stride::Int=1, pad::Int=0) where T
    C, H, W, B = size(x)
    H_out, W_out = _out_dims(H, W, k, stride, pad)
    
    # 预计算输出大小
    n_patches = H_out * W_out * B
    patch_size = C * k * k
    
    # Zygote 对 Array 构造和 setindex! 的支持有限，
    # 这里的写法是为了兼顾性能和 Zygote 的兼容性。
    # 更高效的做法是使用 ChainRules 定义 rrule，但这里我们用 pure Julia 实现。
    
    # 注意：为了让 Zygote 能够求导，我们尽量避免复杂的原地操作，
    # 但对于 im2col 这种重排，Buffer 是必要的。

    # 注意：这里假设 x 是 (C, H, W, B)
    # 如果有 padding，需要先 pad array，这里简化为 pad=0

    # 提取 patch: (C, k, k) -> vec -> (C*k*k)

    col = Zygote.Buffer(Array{T}(undef, patch_size, n_patches))
    
    idx = 1
    for b in 1:B, w in 1:stride:W-k+1, h in 1:stride:H-k+1
        patch = x[:, h:h+k-1, w:w+k-1, b]
        col[:, idx] = vec(patch)
        idx += 1
    end
    return copy(col), H_out, W_out
end




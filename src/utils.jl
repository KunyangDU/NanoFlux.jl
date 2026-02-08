# relu(x::T) where T = max(zero(T), x)
peek(x::AbstractArray;ind::Int64 = 1) = colorview(RGB,clamp01.(permutedims(x[:,:,:,ind], (3, 2,1))))



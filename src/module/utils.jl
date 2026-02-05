
function _fmt_shape(s)
    dims = s[1:end-1] 
    return string(dims)
end

_count_params(l::Any) = 0
_count_params(l::Dense) = length(l.W) + length(l.b)
_count_params(l::Conv)  = length(l.W) + length(l.b)

function format_number(n::Int)
    return replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
end
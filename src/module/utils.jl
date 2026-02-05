
function _fmt_shape(s)
    dims = s[1:end-1] 
    return string(dims)
end

function format_number(n::Int)
    return replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
end


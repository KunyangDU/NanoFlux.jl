"""
    SimpleTokenizer{T}

通用的简单分词器。
T: Token 的类型 (通常是 Char 或 String)。

字段:
- stoi: Token -> ID
- itos: ID -> Token
- splitter: 一个函数，定义如何将长文本切分成 Token 列表
"""
struct SimpleTokenizer{T,L}
    stoi::Dict{T, Int}
    itos::Vector{T}
    splitter::Function # 核心差异点: 切分逻辑
end

SimpleTokenizer(stoi::Dict{T, Int}, itos::Vector{T}, splitter::Function) where T = 
    SimpleTokenizer{T, length(itos)}(stoi, itos, splitter)

"""
    build_tokenizer(text::String; mode=:char)

构建分词器的工厂函数。
- mode=:char : 字符级 (CharTokenizer)
- mode=:word : 单词级 (WordTokenizer, 按空格切分)
"""
function build_tokenizer(text::String; mode=:char)
    if mode == :char
        splitter = s -> collect(s)
        tokens = splitter(text)
        T = Char
    elseif mode == :word
        splitter = s -> split(s) 
        tokens = String.(splitter(text)) # 转为 String
        T = String
    else
        error("Unsupported mode: $mode")
    end

    unique_tokens = sort(unique(tokens))

    stoi = Dict(t => i for (i, t) in enumerate(unique_tokens))
    itos = unique_tokens
    
    return SimpleTokenizer{T,length(itos)}(stoi, itos, splitter)
end

"""
    encode(t::SimpleTokenizer{T}, s::AbstractString)

文本 -> ID 列表
"""
function encode(t::SimpleTokenizer{T}, s::AbstractString) where T
    raw_tokens = t.splitter(s)
    
    if T == String
        raw_tokens = String.(raw_tokens)
    end
    
    return [t.stoi[token] for token in raw_tokens]
end

"""
    decode(t::SimpleTokenizer{T}, indices::AbstractVector{<:Integer})

ID 列表 -> 文本
"""
function decode(t::SimpleTokenizer{T}, indices::AbstractVector{<:Integer}) where T
    tokens = [t.itos[i] for i in indices]
    
    if T == Char
        return String(tokens)
    else
        return join(tokens, " ")
    end
end

vocab_size(::SimpleTokenizer{T,L}) where {T,L} = L

function Base.show(io::IO, t::SimpleTokenizer{T}) where T
    print(io, "SimpleTokenizer{$T}(vocab=$(vocab_size(t)))")
end
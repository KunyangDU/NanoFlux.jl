using Images
using FileIO
using Random
using MLUtils # 必须引用
using DataFrames, CSV
# ---------------------------------------------------------
# 1. 辅助工具：构建词表和分词
# ---------------------------------------------------------
# const PAD_IDX = 1
# const BOS_IDX = 2
# const EOS_IDX = 3
# const UNK_IDX = 4

# function build_vocab(captions::Vector{String}; min_freq=5)
#     freqs = Dict{String, Int}()
#     for cap in captions
#         for w in split(lowercase(cap))
#             freqs[string(w)] = get(freqs, string(w), 0) + 1
#         end
#     end

#     itos = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
#     for (w, c) in freqs
#         if c >= min_freq
#             push!(itos, w)
#         end
#     end
    
#     stoi = Dict(w => i for (i, w) in enumerate(itos))
#     return SimpleTokenizer{String, length(itos)}(stoi, itos, s -> split(lowercase(s)))
# end

# function encode_caption(tokenizer, text::String, max_len::Int,
#      PAD_IDX = 1,
#      BOS_IDX = 2,
#      EOS_IDX = 3,
#      UNK_IDX = 4)
#     words = tokenizer.splitter(text)
#     ids = [get(tokenizer.stoi, w, UNK_IDX) for w in words]
    
#     # 截断留位给 BOS/EOS
#     if length(ids) > max_len - 2
#         ids = ids[1:max_len-2]
#     end
    
#     # 添加首尾 Tag
#     ids = [BOS_IDX; ids; EOS_IDX]
    
#     # Padding
#     pad_len = max_len - length(ids)
#     if pad_len > 0
#         append!(ids, fill(PAD_IDX, pad_len))
#     end
#     return ids
# end

struct FlickrDataset <: AbstractDataset
    img_dir::String
    metadata::DataFrame
    tokenizer::SimpleTokenizer
    transform::Function
end

Base.length(d::FlickrDataset) = nrow(d.metadata)

function Base.getindex(d::FlickrDataset, i::Int)
    row = d.metadata[i, :]
    
    img_path = joinpath(d.img_dir, row.image_id)
    
    img = try
        load(img_path)
    catch e
        @warn "Image load failed: $img_path"
        
        # [关键修复] 返回 2x2 的黑色图片
        # imresize 的 Linear 插值算法要求输入维度至少为 2
        fill(RGB{N0f8}(0,0,0), 2, 2)
    end
    
    x_img = d.transform(img)
    y_tokens = row.token_ids
    
    return (x_img, y_tokens)
end

# 批量索引支持
Base.getindex(d::FlickrDataset, idxs::AbstractVector) = [d[i] for i in idxs]

# MLUtils 适配
MLUtils.getobs(d::FlickrDataset, i) = d[i]
MLUtils.numobs(d::FlickrDataset) = length(d)

# ==============================================================================
# 3. 加载逻辑 (Loading Logic)
# ==============================================================================

function load_flickr8k(img_dir::String, token_file::String, split_file::String; max_len=40, min_freq=5)
    println("Loading Metadata...")

    # 1. 读取 CSV
    df = CSV.read(token_file, DataFrame, header=false, delim='\t', silencewarnings=true)
    rename!(df, [:id_code, :caption])
    
    # 2. 清洗 Image ID
    df.image_id = [split(x, '#')[1] for x in df.id_code]
    
    # 3. 筛选训练集
    train_imgs = Set(readlines(split_file))
    filter!(row -> row.image_id in train_imgs, df)
    
    # 4. 清洗空数据
    dropmissing!(df, :caption)
    filter!(row -> !isempty(row.caption), df)
    println("Found $(nrow(df)) training captions.")

    # 5. 构建词表
    println("Building Tokenizer...")
    tokenizer = build_tokenizer(String.(df.caption), min_freq=min_freq)
    println("Vocab Size: $(length(tokenizer.stoi))")

    # 6. Tokenize (无需传递 idx 参数，使用默认值)
    println("Tokenizing all captions...")
    df.token_ids = [encode_text(tokenizer, row.caption, max_len) for row in eachrow(df)]

    # 7. 定义默认图片变换
    default_transform(img) = begin
        img_resized = imresize(img, (224, 224))
        # 确保输出是 (W, H, C)
        return Float32.(permutedims(channelview(img_resized), (3, 2, 1)))
    end

    return FlickrDataset(img_dir, df, tokenizer, default_transform)
end


# ==============================================================================
# 1. 扩展 Tokenizer (不重新定义 struct)
# ==============================================================================

# 注意：这里不再定义 struct SimpleTokenizer，因为 tokenizer.jl 已经定义了。
# 我们只需要增加一个新的方法 (Multiple Dispatch) 来处理 Vector{String}。

"""
    build_tokenizer(captions::AbstractVector{<:AbstractString}; min_freq=5)

重载 build_tokenizer，使其支持字符串数组（针对 Flickr8k 数据集）。
"""
function build_tokenizer(captions::AbstractVector{<:AbstractString}; min_freq=5)
    freqs = Dict{String, Int}()
    splitter = s -> split(lowercase(s))
    
    # 统计词频
    for cap in captions
        for w in splitter(cap)
            freqs[string(w)] = get(freqs, string(w), 0) + 1
        end
    end

    # 特殊 Token (顺序: PAD=1, BOS=2, EOS=3, UNK=4)
    # 必须全是 String 类型以匹配 SimpleTokenizer{String}
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    itos = copy(special_tokens)
    
    for (w, c) in freqs
        if c >= min_freq
            push!(itos, w)
        end
    end
    
    stoi = Dict(w => i for (i, w) in enumerate(itos))
    
    # 构造原有的参数化结构体 SimpleTokenizer{T, L}
    # T = String, L = 词表大小
    return SimpleTokenizer{String, length(itos)}(stoi, itos, splitter)
end

"""
    encode_text(t, text, max_len; ...)

[隐藏接口] 使用默认关键字参数处理特殊 Token ID。
"""
function encode_text(t::SimpleTokenizer, text::String, max_len::Int; 
                     pad_idx=1, bos_idx=2, eos_idx=3, unk_idx=4)
    words = t.splitter(text)
    
    # 查找 ID
    ids = [get(t.stoi, w, unk_idx) for w in words]
    
    # 截断
    if length(ids) > max_len - 2
        ids = ids[1:max_len-2]
    end
    
    # 添加 BOS(2) 和 EOS(3)
    ids = [bos_idx; ids; eos_idx]
    
    # Padding(1)
    pad_len = max_len - length(ids)
    if pad_len > 0
        append!(ids, fill(pad_idx, pad_len))
    end
    
    return ids
end
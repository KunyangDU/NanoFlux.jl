include("../src/NanoFlux.jl")

dataname = "examples/dataset/flickr8k"

# 假设文件路径
img_dir = "$(dataname)/Flickr8k_Dataset" # 图片文件夹
token_file = "$(dataname)/Flickr8k_text/Flickr8k.token.txt"
train_split = "$(dataname)/Flickr8k_text/Flickr_8k.trainImages.txt"

# 1. 加载
dataset = load_flickr8k(img_dir, token_file, train_split; max_len=40)

# 2. 创建 DataLoader (现在不会报错了)
# collate=true 会自动把 Vector{Tuple} 堆叠成 Batch Matrix
train_loader = DataLoader(dataset, batchsize=32, shuffle=true, collate=true)

# 3. 测试
x, y = first(train_loader)
println("Image Batch: ", size(x)) # 应为 (224, 224, 3, 32)
println("Text Batch: ", size(y))  # 应为 (40, 32)
peek(x)

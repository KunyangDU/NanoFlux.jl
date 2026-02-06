include("../src/NanoFlux.jl")

BATCH_SIZE = 64    # 每一批处理多少个序列
BLOCK_SIZE = 128    # 上下文长度 (Time steps)
EMBED_DIM  = 128   # 嵌入维度
N_HEADS    = 4     # 注意力头数
N_LAYERS   = 3     # Block 层数
LR         = 1e-3  # 学习率 (Adam 默认)
EPOCHS     = 1    # 训练轮数
TRAIN_DATA = "shakespeare.txt"
CUT_STEP   = 1000

text = read("examples/dataset/$(TRAIN_DATA)",String)
# text = repeat(text, 100)
tokenizer = build_tokenizer(text, mode=:char)
VOCAB_SIZE = vocab_size(tokenizer)
data_ids = encode(tokenizer, text)
dataset = CausalLMDataset(data_ids, BLOCK_SIZE)
loader = DataLoader(dataset, batchsize=BATCH_SIZE, shuffle=true)

model = Sequential(
    Embed(VOCAB_SIZE, EMBED_DIM),
    Position(EMBED_DIM, BLOCK_SIZE),
    [Block(EMBED_DIM, N_HEADS, BLOCK_SIZE) for _ in 1:N_LAYERS]...,
    LayerNorm(EMBED_DIM),
    Dense(EMBED_DIM, VOCAB_SIZE) 
)
optimizer = Adam(learning_rate=Float32(LR))
config = TrainerConfig(
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    show_times = 10,
    target_loss = 0.1,
    target_acc = 0.98,
    cut_step = CUT_STEP
)
summary(model,(BLOCK_SIZE,))
ps, history = let ps = initialize(model)
    train!(model, ps, loader, optimizer, config)
end

println("-"^40)
println("Generative Test:")
prompt = "To be"
generated = generate(model, ps, tokenizer, prompt, max_new_tokens=100, block_size=BLOCK_SIZE)
println("Input: $prompt")
println("Output: ", repr(generated))
println("-"^40)

@save "examples/data/ps_T$(TRAIN_DATA)_C$(CUT_STEP)_B$(BATCH_SIZE)_T$(BLOCK_SIZE)_D$(EMBED_DIM)_H$(N_HEADS)_L$(N_LAYERS)_lr$(LR)_ep$(EPOCHS).jld2" ps
@save "examples/data/history_T$(TRAIN_DATA)_C$(CUT_STEP)_B$(BATCH_SIZE)_T$(BLOCK_SIZE)_D$(EMBED_DIM)_H$(N_HEADS)_L$(N_LAYERS)_lr$(LR)_ep$(EPOCHS).jld2" history

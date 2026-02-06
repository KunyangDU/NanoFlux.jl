
include("../src/NanoFlux.jl")

function mnist_loader(batch_size::Int; split=:train)
    raw_x, raw_y = MNIST(split=split)[:]
    dataset = SpatialDataset(raw_x, raw_y; num_classes=10, add_channel_dim=true)
    do_shuffle = (split == :train)
    loader = DataLoader(dataset, batchsize=batch_size, shuffle=do_shuffle)
    return loader
end

model = Sequential(
    Conv(2, 1, 6, 5; stride=1, act=relu),
    Pool(2, 2; stride=2, mode=maximum),
    Conv(2, 6, 16, 5; stride=1, act=relu),
    Pool(2, 2; stride=2, mode=maximum),
    Flatten(),
    Dense(256, 120, relu),
    Dense(120, 84, relu),
    Dense(84, 10, identity)
)

# opt = SGD(
#     learning_rate = 0.001, 
#     momentum = 0.9
# )
opt = Adam(
    learning_rate = 0.001, 
)
config = TrainerConfig(
    epochs = 5,
    batch_size = 64,
    show_times = 100,
    target_loss = 0.1,
    target_acc = 0.98,
    patience = 1,
)
summary(model,(28,28,1))

train_loader = mnist_loader(config.batch_size)

ps,history = let ps = initialize(model)
    train!(model, ps, train_loader, opt, config)
end



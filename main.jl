
include("src/NanoFlux.jl")

algo = SimpleAlgorithm(16,5,0.1,0.9,5)

model = Sequential([
    Convolution(1,4,(28,28),2),
    Pooling(2, 4, (27, 27)),
    Dense(676, 128),
    Dense(128, 10, identity)
])

loader = mnist_loader(algo.batch_size)

train!(model,loader,algo)





using MLDatasets, MLUtils, OneHotArrays
using Zygote, NNlib
using Random
using Printf, TimerOutputs
using Metal
using Statistics: mean

include("default.jl")

include("abstract.jl")

include("wrapper/tensor.jl")

include("control/algorithm.jl")
include("control/information.jl")

include("module/sequential.jl")
include("module/dense.jl")
include("module/convolution.jl")
include("module/pool.jl")
include("module/flatten.jl")
include("module/input.jl")
include("module/check.jl")
include("module/utils.jl")
include("module/initialize.jl")

include("fileIO/utils.jl")
include("fileIO/mnist.jl")

include("algorithm/train.jl")
include("algorithm/update.jl")
include("algorithm/loss.jl")

include("optimizer/SGD.jl")
include("optimizer/Adam.jl")

include("wrapper/interface.jl")

include("fileIO/utils.jl")
include("fileIO/mnist.jl")
include("utils.jl")
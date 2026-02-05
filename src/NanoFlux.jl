using MLDatasets, MLUtils, OneHotArrays
using Zygote, Tullio
using ChainRulesCore, ForwardDiff
using Printf, TimerOutputs
using Metal, LoopVectorization
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

include("fileIO/utils.jl")
include("fileIO/mnist.jl")

include("algorithm/train.jl")

include("wrapper/interface.jl")

include("fileIO/utils.jl")
include("fileIO/mnist.jl")
include("utils.jl")
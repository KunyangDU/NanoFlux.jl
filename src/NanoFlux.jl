using MLDatasets, MLUtils, OneHotArrays
using Zygote, TensorKit
using ChainRulesCore
using Printf

include("abstract.jl")
include("patch.jl")

include("control/algorithm.jl")
include("control/information.jl")

include("layer/sequential.jl")
include("layer/dense.jl")

include("fileIO/utils.jl")
include("fileIO/mnist.jl")

include("algorithm/forward.jl")
include("algorithm/train.jl")


include("fileIO.jl")
include("utils.jl")
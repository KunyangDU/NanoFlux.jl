using MLDatasets, MLUtils, OneHotArrays
using Zygote, NNlib
using Random
using Printf, TimerOutputs
using Statistics: mean, var
using JLD2
using LinearAlgebra: triu, dot

include("abstract.jl")

include("default.jl")

include("wrapper/tensor.jl")

include("control/algorithm.jl")
include("control/information.jl")

include("module/Identity.jl")
include("module/sequential.jl")
include("module/dense.jl")
include("module/convolution.jl")
include("module/pool.jl")
include("module/flatten.jl")
# include("module/input.jl")
include("module/attention.jl")
include("module/normalize.jl")
include("module/block.jl")
include("module/embed.jl")
include("module/utils.jl")
include("module/initialize.jl")
include("module/summary.jl")
include("module/show.jl")
include("module/resnet.jl")
include("module/time.jl")
include("module/spatial.jl")

include("network/UNetAttentionBlock.jl")
include("network/UNet.jl")

include("algorithm/train.jl")
include("algorithm/update.jl")
include("algorithm/loss.jl")
include("algorithm/generate.jl")
include("algorithm/diffusion.jl")

include("optimizer/SGD.jl")
include("optimizer/Adam.jl")

include("wrapper/interface.jl")

include("fileIO/utils.jl")
include("fileIO/tokenizer.jl")
include("fileIO/lm.jl")
include("fileIO/spatial.jl")

include("gpu/move.jl")

include("utils.jl")
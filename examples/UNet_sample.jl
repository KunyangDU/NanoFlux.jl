include("../src/NanoFlux.jl")
using ImageIO,ImageCore

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
CUT_STEP = Inf
BASE_CH = 32  
TIME_DIM = BASE_CH * 4 
TIME_STEP = 400

@load "examples/data/ddpm_model_CIFAR10_T$(TIME_STEP)_CH$(BASE_CH)_B$(BATCH_SIZE)_LR$(LEARNING_RATE)_EP$(EPOCHS)_Cut$(CUT_STEP).jld2.jld2" model
@load "examples/data/ddpm_ps_CIFAR10_T$(T)_CH$(BASE_CH)_B$(BATCH_SIZE)_LR$(LEARNING_RATE)_EP$(EPOCHS)_Cut$(CUT_STEP).jld2.jld2" ps
dconfig = DiffusionProcess(TIME_STEP)

imgs = sample(model, initialize(model), dconfig, (32, 32, 3, 1))[1]


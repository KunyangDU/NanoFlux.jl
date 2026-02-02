function train!(model::Sequential,train_loader::DataLoader, algo::SimpleAlgorithm)
    info = SimpleInformation()

    info.velocities = _initial_velocities(model)

    for _ in 1:algo.epochs
        count = 1
        L = length(train_loader)
        for loader in train_loader
            x, y = batch_to_tensormap(loader...)
            _train!(model,x,y,algo,info)
            if mod(count,algo.show_times) == 0
                print("$(count)/$(L) - ")
                show(info)
            end
            count += 1
        end
        println("-"^38)
        show(info)
        println("-"^38)
    end
    return info
end


function _train!(model::Sequential, x::AbstractTensorMap, y::AbstractTensorMap, algo::SimpleAlgorithm, info::SimpleInformation)
    ps = _params(model)
    
    loss_val, gs = Zygote.withgradient(ps) do
        loss(model, x, y)
    end
    
    for p in ps
        if gs[p] !== nothing
            Δ = gs[p] isa TensorMap ? gs[p].data : gs[p]
            v_data = info.velocities[p].data

            @. v_data = algo.momentum * v_data + Δ
            @. p.data -= algo.learning_rate * v_data
        end
    end

    push!(info.loss,loss(model,x,y))
    push!(info.accuracy,accuracy(model,x,y))
    
    return loss_val
end


function _params(model::Sequential)
    ps = Zygote.Params()
    for layer in model.layers
        _params(ps,layer)
    end
    return ps
end

function _params(ps::Params, layer::Dense)
    push!(ps, layer.W)
    push!(ps, layer.b)
end

function loss(A::Sequential, x::AbstractTensorMap, y::AbstractTensorMap)
    logits_tm = forward(A,x)
    
    logits = convert(Array, logits_tm)
    y_target = convert(Array, y)

    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    L = -sum(y_target .* log.(probs .+ 1e-10)) / size(logits, 2)
    
    return L
end

function accuracy(A::Sequential, x::AbstractTensorMap, y::AbstractTensorMap)
    logits_tm = forward(A,x)
    logits = convert(Array, logits_tm)
    targets = convert(Array, y)
    y_pred = map(c -> c[1], argmax(logits, dims=1))
    y_true = map(c -> c[1], argmax(targets, dims=1))
    return sum(y_pred .== y_true)/length(y_true)
end


function _initial_velocities(model::Sequential)
    velocities = IdDict()
    params = _params(model)
    for p in params
        velocities[p] = p * 0.0 
    end
    return velocities
end


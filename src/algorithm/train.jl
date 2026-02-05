
function train!(model::AbstractModule, train_loader, algo::SimpleAlgorithm)

    history = TrainingHistory()
    velocities = _initial_velocities(model)
    manualGC()

    total_loaders = length(train_loader)

    for epoch in 1:algo.epochs
        for (x_raw, y_raw) in train_loader

            @timeit TO "Data Prepare" begin
                ndims_spatial = ndims(x_raw) - 2
                x = SpatialTensor{ndims_spatial}(x_raw)
                y = y_raw
            end

            @timeit TO "Back Propagation" begin
                loss_val = _train_step!(model, x, y, algo, velocities, history)
            end

            if algo.show_times > 0 && mod(mod(history.count - 1, total_loaders) + 1, algo.show_times) == 0
                print("Epoch $(epoch) [$(mod(history.count-1, total_loaders) + 1)/$(total_loaders)] - ")
                show(history)
            end

            history.count += 1
        end 
        history.avg_loss = mean(history.loss[end - length(train_loader) + 1:end])
        history.avg_acc = mean(history.accuracy[end - length(train_loader) + 1:end])

        @timeit TO "gc" manualGC()

        show(TO;title = "$(epoch) / $(algo.epochs)")
        print("\n")
        show(history)

        if algo.target_loss !== nothing && history.avg_loss  <= algo.target_loss
            if history.count_loss ≥ algo.patience
                println()
                println(bg"Target Loss Reached!"," ($(history.avg_loss) <= $(algo.target_loss))")
                println("Stopping training early at Epoch $epoch.")
                break
            else
                history.count_loss += 1
            end
        end

        if algo.target_acc !== nothing && history.avg_acc >= algo.target_acc
            if history.count_acc ≥ algo.patience
                println()
                println(bg"Target Accuracy Reached!"," ($(history.avg_acc) >= $(algo.target_acc))")
                println("Stopping training early at Epoch $epoch.")
                break
            else
                history.count_acc += 1
            end
        end
    end
    
    show(TO)
    print("\n")
    
    return history
end


function _train_step!(model::AbstractModule, x, y, 
                      algo::SimpleAlgorithm, 
                      velocities::IdDict, 
                      history::TrainingHistory)
    
    ps = _params(model)

    loss_val, gs = @timeit TO "calc gradient" Zygote.withgradient(ps) do
        loss(model, x, y)
    end

    @timeit TO "update" for p in ps
        if gs[p] !== nothing
            g = gs[p]

            v = get!(velocities, p, zeros(eltype(p), size(p)))

            @. v = algo.momentum * v + g
            @. p -= algo.learning_rate * v
        end
    end

    if isdefined(history, :loss)
        push!(history.loss, loss_val)
        push!(history.accuracy, accuracy(model, x, y))
    end
    
    return loss_val
end

function _initial_velocities(model::AbstractModule)
    velocities = IdDict()
    ps = _params(model)
    for p in ps
        velocities[p] = zeros(eltype(p), size(p))
    end
    return velocities
end

function loss(model::AbstractModule, x, y)
    y_pred = model(x)
    logits = y_pred.data
    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    return -sum(y .* log.(probs .+ 1e-10)) / size(logits, 2)
end

function accuracy(model::AbstractModule, x, y)
    y_pred = model(x)
    logits = y_pred.data
    pred_idx = [c[1] for c in argmax(logits, dims=1)]
    true_idx = [c[1] for c in argmax(y, dims=1)]
    return mean(pred_idx .== true_idx)
end

"""
    params(m::AbstractModule)

返回一个 Zygote.Params 对象，包含该模块及其子模块的所有可训练参数。
"""
function _params(m::AbstractModule)
    ps = Params()
    _collect_params!(ps, m)
    return ps
end

function _collect_params!(ps::Params, m::Sequential)
    for layer in m.layers
        _collect_params!(ps, layer)
    end
end

function _collect_params!(ps::Params, m::AbstractModule)
    if hasfield(typeof(m), :W)
        push!(ps, m.W)
    end
    if hasfield(typeof(m), :b)
        push!(ps, m.b)
    end
    # 其他参数
end
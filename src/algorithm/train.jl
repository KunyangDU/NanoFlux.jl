
function train!(model::AbstractModule, ps::ParamsContainer,
                    train_loader::DataLoader, 
                    opt::AbstractOptimizer, 
                    config::TrainerConfig)
    history = TrainingHistory()
    opt_state = to_device(initialize(opt, ps))
    manualGC()

    total_loaders = length(train_loader)
    last_show_time = time()
    for epoch in 1:config.epochs
        history.count > config.cut_step && break
        for (x_raw, y_raw) in train_loader
            history.count > config.cut_step && break
            x_raw = to_device(x_raw)
            y_raw = to_device(y_raw)
            
            @timeit TO "Data Prepare" begin
                ndims_spatial = ndims(x_raw) - 2
                x = SpatialTensor{ndims_spatial}(x_raw)
                y = y_raw
            end

            @timeit TO "Back Propagation" _train_step!(model, x, y, ps, opt_state, opt, history, config.config)

            if config.show_times > 0 && mod(mod(history.count - 1, total_loaders) + 1, config.show_times) == 0
                current_time = time()
                elapsed = current_time - last_show_time
                print("Epoch $(epoch) [$(mod(history.count-1, total_loaders) + 1)/$(total_loaders)] - ")
                @printf("%.2fs - ", elapsed)
                show(history)
                last_show_time = current_time
            end

            history.count += 1
        end 
        history.avg_loss = mean(history.loss[max(1, length(history.loss) - length(train_loader) + 1):end])
        history.avg_acc = mean(history.accuracy[max(1, length(history.accuracy) - length(train_loader) + 1):end])

        @timeit TO "gc" manualGC()

        show(TO;title = "$(epoch) / $(config.epochs)")
        print("\n")
        show(history)

        if config.target_loss !== nothing && history.avg_loss  <= config.target_loss
            if history.count_loss ≥ config.patience
                println()
                println(bg"Target Loss Reached!"," ($(history.avg_loss) <= $(config.target_loss))")
                println("Stopping training early at Epoch $epoch.")
                break
            else
                history.count_loss += 1
            end
        end

        if config.target_acc !== nothing && history.avg_acc >= config.target_acc
            if history.count_acc ≥ config.patience
                println()
                println(bg"Target Accuracy Reached!"," ($(history.avg_acc) >= $(config.target_acc))")
                println("Stopping training early at Epoch $epoch.")
                break
            else
                history.count_acc += 1
            end
        end
    end
    
    show(TO)
    print("\n")
    
    return ps,history
end


function _train_step!(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, 
                        ps::ParamsContainer,
                        opt_state::ParamsContainer,
                        opt::AbstractOptimizer,
                        history::TrainingHistory,
                        config::AbstractAlgorithm)

    @timeit TO "calc gradient" loss_val, grads = Zygote.withgradient(p -> loss(model, x, y, p, config), ps)

    @timeit TO "update!" update!(ps, grads[1], opt_state, opt)

    if isdefined(history, :loss)
        push!(history.loss, loss_val)
        push!(history.accuracy, accuracy(model, x, y, ps, config))
    end

end



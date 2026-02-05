
function train!(model::AbstractModule, train_loader, algo::SimpleAlgorithm)
    # 1. 初始化基础设施
    history = TrainingHistory()

    # 3. 初始化动量缓存 (Velocities)
    # 这一步也需要适配 AbstractModule，下面有定义
    velocities = _initial_velocities(model)
    
    # 4. 垃圾回收 (清理之前的内存)
    manualGC()

    total_batches = try length(train_loader) catch; 0 end
    for epoch in 1:algo.epochs
        count = 1
        # 进度打印辅助
        for (x_raw, y_raw) in train_loader
            
            # --- A. 数据准备 ---
            @timeit TO "Data Prepare" begin
                # 自动识别维度并包装 (适配 Conv 和 Dense)
                ndims_spatial = ndims(x_raw) - 2
                x = SpatialTensor{ndims_spatial}(x_raw)
                y = y_raw
            end

            # --- B. 核心训练 (Forward + Backward + Update) ---
            @timeit TO "Back Propagation" begin
                loss_val = _train_step!(model, x, y, algo, velocities, history)
            end

            # --- C. 日志打印 ---
            if algo.show_times > 0 && mod(count, algo.show_times) == 0
                print("Epoch $(epoch) [$(count)/$(total_batches)] - ")
                show(history)
            end
            count += 1
        end        
        @timeit TO "gc" manualGC()

        show(TO;title = "$(epoch) / $(algo.epochs)")
        print("\n")
        show(history)
    end
    
    show(TO)
    
    return history
end

# ==============================================================================
# 3. 单步训练 (Step)
# ==============================================================================

function _train_step!(model::AbstractModule, x, y, 
                      algo::SimpleAlgorithm, 
                      velocities::IdDict, 
                      history::TrainingHistory)
    
    # 1. 获取参数 (使用通用的 params 接口)
    # Zygote 会智能处理，这一步开销很小
    ps = _params(model)
    
    # 2. 计算梯度
    loss_val, gs = @timeit TO "calc gradient" Zygote.withgradient(ps) do
        loss(model, x, y)
    end
    
    # 3. 参数更新 (Momentum SGD)
    # 注意：这里我们遍历的是 ps (参数集合)，而不是 layers
    # 这意味着无论网络结构多复杂，更新逻辑都是统一的
    @timeit TO "update" for p in ps
        if gs[p] !== nothing
            g = gs[p]
            
            # 从缓存中取出对应的速度，如果没有就初始化为0
            # (这是防止 _initial_velocities 漏掉某些参数的安全网)
            v = get!(velocities, p, zeros(eltype(p), size(p)))
            
            # 动量更新 (In-place)
            @. v = algo.momentum * v + g
            @. p -= algo.learning_rate * v
        end
    end

    # 4. 记录日志
    if isdefined(history, :loss)
        push!(history.loss, loss_val)
        push!(history.accuracy, accuracy(model, x, y))
    end
    
    return loss_val
end

# ==============================================================================
# 4. 辅助函数：初始化动量
# ==============================================================================

function _initial_velocities(model::AbstractModule)
    velocities = IdDict()
    ps = _params(model) # 复用通用的 params
    for p in ps
        velocities[p] = zeros(eltype(p), size(p))
    end
    return velocities
end

# ==============================================================================
# 5. Loss 和 Accuracy (保持不变，但签名改为 AbstractModule)
# ==============================================================================

function loss(model::AbstractModule, x, y)
    y_pred = model(x) # 调用 AbstractModule 的前向传播
    # ... (具体的 CrossEntropy 逻辑同前) ...
    # 为了完整性简写如下：
    logits = y_pred.data
    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    return -sum(y .* log.(probs .+ 1e-10)) / size(logits, 2)
end

function accuracy(model::AbstractModule, x, y)
    y_pred = model(x)
    logits = y_pred.data
    pred_idx = [c[1] for c in argmax(logits, dims=1)]
    # 假设 y 是 One-Hot
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
    # 检查是否有 W 字段
    if hasfield(typeof(m), :W)
        push!(ps, m.W)
    end
    # 检查是否有 b 字段
    if hasfield(typeof(m), :b)
        push!(ps, m.b)
    end
    # 如果未来有其他参数 (比如 Gamma, Beta)，也可以加在这里
end
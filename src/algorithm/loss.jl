function loss(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer)
    y_pred = model(x, ps)
    logits = y_pred.data
    logits_safe = logits .- maximum(logits, dims=1)
    probs = exp.(logits_safe) ./ sum(exp.(logits_safe), dims=1)
    return -sum(y .* log.(probs .+ 1e-10)) / size(logits, 2)
end

function accuracy(model::AbstractModule, x::AbstractNanoTensor, y::AbstractArray, ps::ParamsContainer)
    y_pred = model(x, ps)
    logits = y_pred.data
    pred_idx = [c[1] for c in argmax(logits, dims=1)]
    true_idx = [c[1] for c in argmax(y, dims=1)]
    return mean(pred_idx .== true_idx)
end
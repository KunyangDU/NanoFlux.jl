
function update!(ps::ParamsContainer, gs::ParamsContainer, states::ParamsContainer, opt::AbstractOptimizer)
    map(ps, gs, states) do p, g, s
        update!(p, g, s, opt)
        if s isa AbstractState 
            s.t += 1
            # opt.learning_rate * (s.t - 1)/s.t
        end
    end
    return nothing
end

update!(::NamedTuple{(), Tuple{}}, ::Any, ::Any, opt::AbstractOptimizer) = nothing
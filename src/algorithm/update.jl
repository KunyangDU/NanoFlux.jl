
function update!(ps::ParamsContainer, gs::ParamsContainer, states::ParamsContainer, opt::AbstractOptimizer)
    map(ps, gs, states) do p, g, s
        update!(p, g, s, opt)
        s isa AbstractState && (s.t += 1)
    end
    return nothing
end

update!(::NamedTuple{(), Tuple{}}, ::Any, ::Any, opt::AbstractOptimizer) = nothing
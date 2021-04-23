using LinearAlgebra
import StatsBase.sample

const Dic = AbstractDict
abstract type AbstractGBM{T, I} end


"""
    Response structure, including spin configuration (:states) and energies.
"""
mutable struct Response{T, I} <: AbstractGBM{T, I}
    W::Matrix{T}
    b::Matrix{T}
    states::Vector{Vector{I}}
    energies::Vector{T}
    indices::Vector
    sample::Dict
end


"""
    Response(n_unit::Int)

Construct response structure with number of unit `n_unit`.

 * T - type of parameters [interacts and biases;] (default: Float64)
 * I - type of spin states (default: Int64)
"""
function Response(n_unit::Int64)
    Response{Float64, Int64}(
        rand(n_unit, n_unit),
        rand(n_unit, 1),
        Vector{Vector}[],
        [],
        [],
        Dict()
    )
end


# initial setting

"""
    create_response(linear::AbstractDict, quad::AbstractDict)

Configure the response structure.
"""
function create_response(linear::Dic, quad::Dic)

    Jkeys = collect.(keys(quad))
    key_list = collect(Set(vcat(keys(linear)..., Jkeys...)))
    elements_match(key_list)
    sort!(key_list)
    linear, quad = relabel_dict(key_list, linear, quad)

    # create field & response
    n_unit = length(key_list)
    resp = Response(n_unit)
    field = zeros((n_unit, 1))
    field[collect(keys(linear))] .= values(linear)

    copy!(resp.W, trans(quad, n_unit))
    copy!(resp.b, field)
    resp.indices = key_list
    return resp
end


"""
    relabel_dict(key_lst::Vector, linear::AbstractDict, quad::AbstractDict)

Replace the keys of the dictionary to unit numbers.
"""
function relabel_dict(key_lst::Vector, linear::Dic, quad::Dic)

    num_nodes = collect(1:length(key_lst))
    tab = Dict(zip(sort(key_lst), num_nodes))
    bias, interact = Dict(), Dict()

    for key in keys(quad)
        key1, key2 = key
        interact[(tab[key1], tab[key2])] = quad[key]
    end

    for key in keys(linear)
        bias[tab[key]] = linear[key]
    end

    return bias, interact
end

# annealing algorithm

"""
    get_energy(resp::Response{T}, state::Vector)

Calculate the energy in the input `state`.
"""
function get_energy(resp::Response{T}, state::Vector) where T
    E = -dot(vec(resp.b), vec(state)) # linear term
    E -= state' * triu(resp.W, 1) * state * 0.5 # interaction term
    return E
end


"""
    accept_rate(dE::T, beta::T)

Return the accpetance rate from energy difference `dE`.
"""
function accept_rate(dE::T, beta::T) where T
    return min(1, exp(-beta * dE))
end


"""
    get_neighbor(old_state::Vector)

Return the proposal state calculated from `old_state`.
"""
function get_neighbor(old_state::Vector)
    new_state = copy(old_state)
    idx = sample(1:length(old_state), 1)
    new_state[idx] *= -1 # single spin flip
    return new_state
end


"""
    anneal(linear::AbstractDict, quad::AbstractDict; [, opts])

Solve forward ising problem expressed by `linear` and `quad` biases.

# Arguments
- `linear::Dict`: Linear biases meaning that magnetic field.
- `quad::Dict`: Quadratic biases meaning that spin-spin interaction.


Options that can be provided in the `opts` dictionary:

* `:beta_min` - Initial (minimal) value of inverse temperature. (default: 5.0)
* `:beta_max` - Final (maximal) value of inverse temperature. (default: 15.0)
* `:n_sweep` - Number of division between `beta_min` and `beta_max`. (default: 1000)
* `:n_read` - Number of run repetitions of Simulated Annealing. (default: 1)


# Examples
```jldoctest
julia> using InverseIsing

julia> h = Dict(:a => 1) # Setting field biases.

julia> J = Dict((:a, :b) => -1) # Setting interaction.

julia> response = anneal(h, J)

julia> response.sample
Dict{Symbol,Int64} with 2 entries:
  :a => 1
  :b => -1
```
"""
function anneal(linear::Dic, quad::Dic; opts...)

    ctx = copy(Dict{Symbol, Real}(opts))
    option_checker!(ctx)
    num_reads = ctx[:n_read]
    num_sweeps = ctx[:n_sweep] # number of division between beta_min & beta_max
    beta_min = ctx[:beta_min] # beta_min: initial beta
    beta_max = ctx[:beta_max] # beta_max: final beta

    resp = create_response(linear, quad)
    state = ones(size(resp.W, 1))
    energy = zero(eltype(resp.W))

    for step in 1:num_reads
        @simd for beta in range(beta_min, beta_max, length = num_sweeps)

            # get neighbor
            new_state = get_neighbor(state)

            # check energy level
            energy = get_energy(resp, state)
            dE = get_energy(resp, new_state) - energy

            # MCMC algorithm
            if rand() < accept_rate(dE, beta)
                state = new_state
            end
        end

        # append some info
        push!(resp.states, Int.(state))
        push!(resp.energies, energy)

    end
    optimal_idx = argmin(resp.energies)
    resp.sample = Dict(zip(resp.indices, resp.states[optimal_idx]))
    return resp

end

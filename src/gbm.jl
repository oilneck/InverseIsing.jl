using LinearAlgebra
using Statistics
using LinearAlgebra.BLAS
import StatsBase.percentile

const Mat{T} = AbstractArray{T, 2}
const Tup{T} = Tuple{Mat{T}, Mat{T}}

abstract type AbstractGBM{T, I} end

"""
General Boltzmann Machine, including parameter matrix (element type T)
and a mask matrix (element type I) for decimation.
"""
mutable struct GBM{T, I} <: AbstractGBM{T, I}
    W::Matrix{T}
    b::Matrix{T}
    W_mask::Array{I, 2}
    PLrange::Dict
end

"""
    GBM(n_unit::Int)

Construct GBM with number of unit `n_unit`.

 * T - type of GBM parameters [weights and biases;] (default: Float64)
 * I - type of Mask-matrix for decimation (default: Int64)
"""
function GBM(n_unit::Int)
    GBM{Float64, Int64}(
        rand(n_unit, n_unit),
        rand(n_unit, 1),
        ones(Int64, (n_unit, n_unit)) - Matrix(I, n_unit, n_unit),
        Dict("min" => 0.0, "max" => 0.0)
    )
end


## gradient calculation

"""
    calc_grad(gmb::GBM, X::AbstractArray, ctx::Dict)

Calculate gradeint of log-pseudo-likelihood with respect to GBM parameter.

# References
- A. Hyvärinen, Neural Comput. **18**, 2283 (2006)
"""
function calc_grad(gbm::GBM{T}, X::Mat{T}, ctx::Dict)::Tup{T} where T

    alpha = T(ctx[:alpha])
    beta = T(ctx[:beta])

    dev = X' - tanh.(beta * (gbm.W * X' .+ gbm.b))
    dW = @getarray(ctx, :dW_buf, size(gbm.W), similar(gbm.W))

    # same as: dW = beta .* dev * X / size(X, 1)
    gemm!('N', 'N', beta / size(X, 1), dev, X, T(0.0), dW)
    symmetrize!(dW)
    axpy!(-alpha, sign.(gbm.W), dW)

    # gradient for bias params.
    db = beta * mean(dev, dims=2)

    return (dW, db)
end


"""
    update(gbm::GBM, grads::Tuple, ctx::Dict)

Update GBM parameters with SGD method.

Update weight parameters of GBM using provided tuple `dtheta = (dW, db)`
of parameter gradients. Optimal method is Stochastic Gradient Descent [SGD].
"""
function update(gbm::GBM{T}, grads::Tup{T}, ctx::Dict) where T
    dW, db = grads
    lr = T(ctx[:lr])

    # same as: gbm.W += lr * dW
    axpy!(lr, dW, gbm.W)
    axpy!(lr, db, gbm.b)

    symmetrize!(gbm.W)
    gbm.W .*= gbm.W_mask
end

## Pseudo likelihood calculation

"""
    pseudo_likelihood(gbm::GBM, X::AbstractArray, ctx::Dict)

Calculate pseudo-likelihood[PL] with spin-example data `X`.
"""
function pseudo_likelihood(gbm::GBM{T}, X::Mat{T}, ctx::Dict)::T where T
    alpha = T(ctx[:alpha])
    beta = T(ctx[:beta])
    H = tr(X * gbm.W * X') + sum(X * gbm.b)
    tmp = sum(log.(cosh.(beta * (gbm.W * X' .+ gbm.b))))
    PL = (beta * H - tmp) / size(X, 1)
    return PL - alpha * sum(abs.(gbm.W))
end




"""
    fit(gbm::GBM, X::AbstractArray; delta=1e-1 [, opts])

Fit the weight parameters of GBM to data `X`.

# Arguments
- `delta::Float`: Threshold for thinning out parameters.


Options that can be provided in the `opts` dictionary:

* `:alpha` - Decay rate of the L1 regularization term. (default: 0.01)
* `:beta` - Inverse temperature. (default: 1.0)
* `:lr` - Learning rate. (default: 1.0)
* `:epochs` - Maximum permissible number of step in SGD. (default: 20)
* `:n_iter` - Number of learning repetitions. (default: 100)


# References
- E. Aurell & M. Ekeberg, Phys. Rev. Lett. **108**, 090201 (2012).

# Examples
```jldoctest
julia> using InverseIsing

julia> samples = [1 -1 -1;] # Spin configuration.

julia> model = GBM(3) # Set the number of units.

julia> fit(model, samples)

julia> W = infer(model); output = decode(W)

julia> output
OrderedCollections.OrderedDict{Tuple{Int64,Int64},Int64} with 3 entries:
  (1, 2) => -1
  (1, 3) => -1
  (2, 3) => 1
```
    The above example means that the interaction between (1, 2) and (1, 3) is
    antiferromagnetic bond and only (2, 3) is ferromagnetic bond.

"""
function fit(gbm::GBM{T}, X::Mat; delta::T=1e-1, opts...) where {T, I}

    ctx = copy(Dict{Any, Any}(opts))
    option_checker!(ctx)
    epochs = ctx[:epochs]
    max_step = ctx[:n_iter]

    X_train = trans_type(T, X)
    variable_check(X_train)

    for n in 1:max_step

        old_likelihood = pseudo_likelihood(gbm, X_train, ctx)
        update(gbm, calc_grad(gbm, X_train, ctx), ctx)

        if n % epochs == 0
            gbm.W_mask[abs.(gbm.W) .< delta] .= 0
            gbm.W_mask .*= gbm.W_mask'
        end

        if isapprox(pseudo_likelihood(gbm, X_train, ctx), old_likelihood)
            break
        end
    end
    return ctx
end


### sub-utility functions ###
"""
    coef(gbm::GBM)

Get weight marix of a trained GBM.

# Examples
```jldoctest
julia> model = GBM(3)

julia> coef(model)
3×3 Array{Float64,2}:
 0.86902   0.4058     0.568231
 0.700701  0.0892165  0.399372
 0.224095  0.234413   0.380341
```
"""
function coef(gbm::GBM{T}) where T
    W = copy(gbm.W)
    # translate -0.0 to 0.0
    W[W .== -T(0.0)] .= T(0.0)
    return W
end


"""
    weights(gbm::GBM)

Same as coef. See also: [`coef`](@ref)
"""
weights = coef


"""
    infer(gbm::GBM)

Infer GBM parameter matrix using trained weight.

# Examples
```jldoctest
julia> using InverseIsing

julia> model = GBM(3)

julia> infer(model)
3×3 Array{Int64,2}:
 1  1  1
 1  1  1
 1  1  1

julia> fit(model, [1 -1 -1;])

julia> infer(model)
3×3 Array{Int64,2}:
  0  -1  -1
 -1   0   1
 -1   1   0
```
"""
function infer(gbm::GBM{T, I}) where {T, I}
    pred = heaviside(coef(gbm))
    return convert(Array{I}, pred)
end

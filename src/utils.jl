import LinearAlgebra.triu


#### function ####

"""
    symmetrize(arr::Array)

Symmetrize the matrix. ``A = (A + A') / 2``
"""
function symmetrize!(arr::Array)
    axpby!(0.5, arr', 0.5, arr)
    copy!(arr, Symmetric(arr, :L))
end


"""
    heaviside(X)

Heaviside's step function.
"""
function heaviside(X)
    return (X .> 0.0) - (X .< 0.0)
end


#### macro ####

macro get(dict, key)
    return esc(quote
        if haskey($dict, $key)
            $dict[$key]
        else
            $nothing
        end
    end)
end


macro getarray(dict, key, shape, default_arr)
    return esc(quote
        if (!haskey($dict, $key) || size($dict[$key]) != $shape)
            $dict[$key] = convert(Array, $default_arr)
        end
        $dict[$key]
    end)
end

#### inspection ####

const OPTIONAL_ITEMS = Dict(
    :alpha => 0.01,
    :beta => 1.0,
    :lr => 1.0,
    :epochs => 20,
    :n_iter => 100 # Option used within `fit` and `annal`
    )

const SAMPLING_ITEMS = Dict(
    :n_sweep => 1000,
    :n_read => 1,
    :beta_min => 5.0, # initial beta
    :beta_max => 15.0 # final beta to stop Simulated Annealing.
    )


"""
    option_warner(opts::Dict)

Display warning if unknown option finded.
"""
function option_warner(opts::Dict, items::Dict)

    OPTIONAL_KEYWARDS = keys(items)

    for opt in keys(opts)
        if !in(opt, OPTIONAL_KEYWARDS)
            @warn("Keyward '$opt' is an invalid optional entry, ignoring")
        end
    end
end


"""
    option_setter!(opts::Dict)

Sets options when a specific key is found.
"""
function option_setter!(opts::Dict, items::Dict)

    OPTIONAL_KEYWARDS = keys(items)

    for key in OPTIONAL_KEYWARDS
        get!(opts, key, items[key])
    end
end


function opt_check(opts::Dict, items::Dict)
    option_warner(opts, items)
    option_setter!(opts, items)
end

option_checker!(opts::Dict{Any, Any}) = opt_check(opts, OPTIONAL_ITEMS)
option_checker!(opts::Dict{Symbol, Real}) = opt_check(opts, SAMPLING_ITEMS)


"""
    variable_check(X::Array)

Check if an element is a spin value: {+1, -1}
"""
function variable_check(X::Array)
    bit_matrix = (X .== 1.0) + (X .== -1.0)
    if !all(Bool.(bit_matrix))
        @warn("Elements of the input array must be spin values: {1, -1}")
    end
end


"""
    elements_match(key_list::Vector)

Check that the types of all elements match.
"""
function elements_match(key_list::Vector)
    flag = all(typeof.(key_list) .== typeof(key_list[1]))
    @assert flag "invalid types of linear and quadratic"
end

#### converter ####

"""
    trans_type(dType::DataType, A::AbstarctArray)

Convert type of matrix elements to `dType`.
"""
function trans_type(dType::DataType, A::AbstractArray)
    (eltype(A) != dType) ? map(dType, A) : A
end


"""
    decode(arr::AbstractArray)

Convert from upper triaugular matrix to dictionary type.

# Example
```jldoctest
julia> a = reshape(1:9, 3, 3)

julia> a
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 1  4  7
 2  5  8
 3  6  9

julia> decode(a)
OrderedCollections.OrderedDict{Tuple{Int64,Int64},Int64} with 3 entries:
  (1, 2) => 4
  (1, 3) => 7
  (2, 3) => 8
```
"""
function decode(arr::AbstractArray)
    triu_mask = triu(trues(size(arr)), 1)
    idx = CartesianIndices(arr)[triu_mask]
    decoded_data = Dict(zip(Tuple.(idx), arr[idx]))
    return sort(decoded_data, by = x -> x[1])
end

"""
    trans(dict::AbstractDict, len::Int64)

Convert from dictionary `dict` to adjacency matrix with size `len` × `len`.

# Examples
```jldoctest
julia> d = Dict((1, 2) => 5, (2, 3) => -1)

julia> trans(d, 4)
4×4 Array{Float64,2}:
 0.0   5.0   0.0  0.0
 5.0   0.0  -1.0  0.0
 0.0  -1.0   0.0  0.0
 0.0   0.0   0.0  0.0
```
"""
function trans(dict::AbstractDict, len::Int64)

    arr = zeros((len, len))
    for (key, val) in zip(keys(dict), values(dict))
        arr[key...]  = val
        arr[reverse(key)...] = val # symmetric
    end
    return arr
end

"""
    trans(dict::AbstarctDict)

Convert from dictionary `dict` to adjacency matrix.

# Example
```jldoctest
julia> d = Dict((1, 2) => -10, (2, 3) => -5, (4, 2) => 1)

julia> trans(d)
4×4 Array{Float64,2}:
   0.0  -10.0   0.0  0.0
 -10.0    0.0  -5.0  1.0
   0.0   -5.0   0.0  0.0
   0.0    1.0   0.0  0.0
```
"""
function trans(dict::AbstractDict)
    len = maximum(maximum.(keys(dict)))
    trans(dict, len)
end

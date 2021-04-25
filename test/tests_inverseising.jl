using InverseIsing
using StatsBase:sample
using JSON

function estim_test1()
    samples = [1 -1 -1;-1 1 1]
    model = GBM(size(samples, 2))
    fit(model, samples)
    W_fit = infer(model)
    @test W_fit == [0 -1 -1;-1 0 1;-1 1 0]
    @test decode(W_fit) == Dict((1, 2) => -1, (1, 3) => -1, (2, 3) => 1)
end

function estim_test2()
    n_unit = 5 # Number of units
    rate = 0.9 # zero rate (in all interaction-pairs)
    h = Dict()
    J_idx = [(i,j) for i in 1:n_unit for j in i+1:n_unit]
    len = length(J_idx)
    J_val = Int.(ones(len))
    J_val[sample(1:len, Int(ceil(len * rate)), replace=false)] .= 0
    J = Dict(zip(J_idx, J_val))

    # annealing and create samples
    resp = anneal(h, J, n_read=1000)
    samples = hcat(resp.states...)'

    # inverse ising inference
    model = GBM(n_unit)
    fit(model, samples)
    pred = decode(infer(model))
    @test J == pred
end


@testset "Test estim" begin
    estim_test1()
    estim_test2()
end

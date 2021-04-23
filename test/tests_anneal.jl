using InverseIsing
using Test


function anneal_test1()
    h = Dict(1 => -1)
    J = Dict((1, 2) => 1)
    res = anneal(h, J)
    @test res.states == [[-1, -1]]
    @test res.indices == [1, 2]
    @test res.sample == Dict(1 => -1, 2 => -1)
    @test res.energies == [-1.5]
end

function anneal_test2()
    # with indices
    h = Dict(:a => 1)
    J = Dict((:a, :b) => -1)
    res = anneal(h, J, n_read = 10)
    @test hcat(res.states...) == repeat([1, -1] ,1, 10)
    @test res.indices == [:a, :b]
    @test res.sample == Dict(:a => 1, :b => -1)
    @test res.energies == repeat([-1.5], 10)
end

function anneal_test3()
    # anti-ferromagnetic
    N = 5 # Number of units
    h = Dict(1 => 1)
    J = Dict(((i,i+1) ,-1) for i in 1:N-1)
    res = anneal(h, J, n_read=10)
    @test res.states[1] == [(-1)^(i+1) for i in 1:N]
    @test [res.sample[key] for key in 1:N] == res.states[argmin(res.energies)]
    @test res.indices == collect(1:N)
end


@testset "Test anneal" begin
    anneal_test1()
    anneal_test2()
    anneal_test3()
end

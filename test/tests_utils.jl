using InverseIsing
using LinearAlgebra:Symmetric
using Test

function utils_test1()
    # test function 'decode'
    A = reshape(collect(1:9), 3, 3)
    @test decode(A) == Dict((1,2) => 4, (1,3) => 7, (2,3) => 8)
end

function utils_test2()
    # test function `trans(dict::AbstractDict, len::Int64)`
    d = Dict((1, 2) => 5, (2, 3) => -1)
    A = zeros(4, 4)
    A[1, 2] = 5; A[2, 3] = -1;
    A = Symmetric(A)
    @test trans(d, 4) == A
end

function utils_test3()
    # test function `trans(dict::AbstractDict)`
    d = Dict((1, 2) => -10, (2, 3) => -5, (2, 4) => 1)
    A = zeros(4, 4)
    for key in keys(d)
        A[key...] = d[key]
    end
    A = Symmetric(A)
    @test trans(d) == A
end



@testset "Test utils" begin
    utils_test1()
    utils_test2()
    utils_test3()
end

using InverseIsing
using Test

const tests =[
    "tests_inverseising",
    "tests_anneal",
    "tests_utils"
]

for t in tests
    @testset "Test $t" begin
        include("$t.jl")
    end
end

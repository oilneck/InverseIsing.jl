using InverseIsing
using LinearAlgebra:Symmetric
using Test
using Dates
using PyPlot




y = [] # processing time
x = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 70, 90, 100] # num_unit
for n_unit in x
    samples = ones(1000, n_unit)
    model = GBM(n_unit)
    print("\nn_unit:", n_unit)
    start = time_ns()
    fit(model, samples, delta=0.0)
    push!(y, Float64(time_ns() - start))
    d = decode(infer(model))
    flg = collect(values(d)) == ones(binomial(n_unit, 2))
    print(" -> ", flg)
end
plot(x, y, ".-")
xlabel("num_unit")
ylabel("time")

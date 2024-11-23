## Assume that you are given a linear system
## Ax = y, where A and y is given as follows
## Using gradient descent method find a vector x so that 
## ||Ax-y|| is minimum.

using Distributions
using Random

A,y = begin
    Random.seed!(0)
    randn(10,10), randn(10,1)
end

function fit(X::Matrix{Float64}, 
    y::Vector{Float64}; 
    lr::Float64 = 0.001, 
    max_iter::Int64 = 1000,
    stopping_criterion::Float64 = 1e-2, 
    seed::Int64 = 10)::Vector{Float64}

    Random.seed!(seed)
    n, m = size(X)
    x = randn(m)
    
    for i in 1:max_iter
        gradient = Zygote.gradient(x->0.5 * norm(A*x - y)^2, x)[1]
        x -= lr * gradient

        if norm(gradient) < stopping_criterion
            println("Converged in $i steps")
            return x
        end

        if i % 10000 == 0
            println("The loss is $(norm(A*x - y))")
        end
        
    end

    @warn "Did not converge in $max_iter steps"
    return x
end


function unit_test()
    try
        @assert isapprox(A\y, x, atol = 1e-1)
    catch AssertionError
        @info "You gotto do it again Pal!!, adjust the learning rate and watch the convergence!!!"
        throw("Buddy wrong time")
    end
    @info "Great Success!!!"
    return 1
end

x = fit(A, vec(y), lr=0.01, max_iter=100000000, stopping_criterion=1e-5) # This seems the most reasonable one
unit_test()
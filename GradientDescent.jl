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
    ## Your code here
    x = zeros(size(X, 2)) # Initialize x with zeros of size of number of columns of X
    
    for i in 1:max_iter 
        gradient = 2 * X' * (X * x - y) # Compute the gradient
        x_new = x - lr * gradient # Moving the point

        if sqrt(sum((x_new - x).^2)) < stopping_criterion # if distance between x_new and x is less than stopping criterion
            println("Converged in $i steps")
            return x_new
        end
        x = x_new # If it is not stopping criterion, update x and continue the loop
    end

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

# x = fit(A, vec(y), lr=0.1) # Does not converge (does not print anything)
# x = fit(A, vec(y), lr=0.1, max_iter=100000) # Does not converge (does not print anything)
# x = fit(A, vec(y), lr=0.1, max_iter=100000, stopping_criterion=1e-5) # Does not converge (does not print anything)
# x = fit(A, vec(y), lr=0.01, max_iter=100000, stopping_criterion=1e-5) # Does not converge (does not print anything)
# x = fit(A, vec(y), lr=0.001, max_iter=100000, stopping_criterion=1e-5) # Does not converge (does not print anything)
# x = fit(A, vec(y), lr=0.0001, max_iter=100000, stopping_criterion=1e-5) # Converges in 64304 steps but still throws an error
# x = fit(A, vec(y), lr=0.00001, max_iter=100000, stopping_criterion=1e-5) # Converges in 46891 steps but still throws an error
# x = fit(A, vec(y), lr=0.000001, max_iter=100000, stopping_criterion=1e-5) # Converges in 6127 steps but still throws an error
# x = fit(A, vec(y), lr=0.0000001, max_iter=100000, stopping_criterion=1e-5) # Converges in 1 step (I think this is also an error) but still throws an error
# x = fit(A, vec(y), lr=0.00000001, max_iter=100000, stopping_criterion=1e-5) # Converges in 1 step (I think this is also an error) but still throws an error
x = fit(A, vec(y), lr=0.000001, max_iter=100000, stopping_criterion=1e-5) # This seems the most reasonable one
unit_test()
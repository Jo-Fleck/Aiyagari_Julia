#=
This script solves the Aiyagari model with value function iteration.
It is inspired by the lectures of Arpad Abraham as well as the corresponding
TA sessions at the European University Institute.

Copyright (C) 2018 Johannes Fleck; https://github.com/Jo-Fleck

You may use this code for your own work and you may distribute it freely.
No attribution is required. Please leave this notice and the above URL in the
source code. Thank you.
=#

# FUN WILL NOW COMMENCE

include("./Markovappr.jl")
include("./Savings_Problem_VFI.jl")


using Distributions

global V
global Rho_sd
global a_vector
global w
global r0
global a

# Parameters

mu     = 3     # relative risk aversion
beta   = 0.96  # discount factor
delta  = 0.08  # depreciation rate
alpha  = 0.36  # capital's share of income

# The original productivity process is AR(1) in logs:
# log(s_t) = rho*log(s_t-1) + sigma*sqrt(1-rho^2)*error_t.
# It gets discretized using a Markov chain with a vector of values log_s
# and a transition matrix pi.

rho   = 0.2   # first-order autoregressive coefficient
sigma = 0.4   # unconditional standard deviation of log(s_t)
S     = 7     # number of shock realizations

pi,log_s,pi_sd = My_Markovappr(rho,sigma*sqrt(1-rho^2),3,S)

s = exp.(log_s);    # grid of labor productivity shocks
N = s'*pi_sd;       # aggregate labor supply

# Parameters of the asset grid
M       = 200    # number of asset grid points
a_min   = -2     # exogenous borrowing limit
a_M     = 25     # maximal level of assets

# Tolerance levels
eps_in  = 1e-8           # tolerance level of the inner loop
eps_out = 1e-3           # tolerance level of the outer loop


# Equilibrium interest rate

# Bounds of the interest rate
minrate = -0.01
maxrate = (1 - beta)/beta

# Outer loop: bisection on the interest rate
count = 0
error_K = 1

tic()

while abs.(error_K) > eps_out
    count = count + 1

    # Select a candidate interest rate
    r0 = (maxrate + minrate)/2

    # Find the demand for capital, given the interest rate
    K_demand = ((r0 + delta)/(alpha*N[1]^(1 - alpha)))^(1/(alpha - 1))

    # Find the supply of savings, given the interest rate
    K_supply = My_Savings_Problem_VFI(r0)
    error_K = K_supply - K_demand

    # Print iteration results
    println("Iteration: ", count, "  r0 = ", round(r0,4), "  error_K = ", round(error_K,4) )

    # Update the candidate interest rate
    err_K = error_K[1]
    if err_K > eps_out
        # supply is too high, decrease the interest rate
        maxrate = r0
    elseif err_K < -eps_out
        # supply is too low, increase the interest rate
        minrate = r0
    end
end

toc()


# Collecting results

# Distribution of agents over the state space
dist = reshape(Rho_sd, S, M)

# Optimal asset decision
a_opt = reshape(a_vector, S, M)

# Optimal consumption decision
c = zeros(S,M)
for i_s = 1:S
    c[i_s,:] = w*s[i_s] + (1 + r0)*a - a_opt[i_s,:]
end

# Make some plots
using PyPlot

figure(1)
plot(a, sum(dist,1)')
xlabel("Assets")
title("Unconditional Distribution of Assets")

figure(2)
plot(a, dist[1,:]/pi_sd[1], label="Lowest labor income")
plot(a, dist[S,:]/pi_sd[S], label="Highest labor income")
xlabel("Assets")
legend()
title("Conditional distribution of assets")

figure(3)
plot(a, c[1,:], label="Lowest labor income")
plot(a, c[S,:], label="Highest labor income")
xlabel("Assets")
ylabel("Consumption")
legend()
title("Optimal consumption")

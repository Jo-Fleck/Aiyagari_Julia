function My_Savings_Problem_VFI(r)

    #=
    This function finds an aggregate supply of assets, given the interest rate
    r using discrete value function iteration.
    =#

# FUN WILL NOW COMMENCE

    global V
    global Rho_sd
    global a_vector
    global w
    global r0
    global a

    # Find equilibrium wage, given the interest rate
    w = (1 - alpha)*((r + delta)/alpha)^(alpha/(alpha - 1))

    # Set lower bound on assets consistently with natural borrowing limit
    if r > 0
        a_1 = max(a_min , - w*s[1]/r)
    else
        a_1 = a_min
    end

    # Create the asset grid
    a = linspace(a_1, a_M, M) # asset grid

    # Value function iteration

    # If this is the first guess of the interest rate, initialize the value
    # function with zeros. Otherwise, initialize it with the value from the
    # previous interest rate guess.
    if count == 1
        V = zeros(S,M)
    end
    U_matrix = zeros(S,M,M)

    # Compute the matrix of utility levels, given the state and the asset choice
    for i = 1:S

        for j = 1:M
            tmp1 = w*s[i] + (1 + r)*a[j]
            tmp_vec = tmp1*ones(M,1)
            tmp2 = tmp_vec - a
            c = max.(tmp2, 1e-6)
            U_matrix[i,j,:] = (c.^(1 - mu) - 1)/(1 - mu)
        end

    end

    # Inner loop: value function iteration
    TV       = zeros(S,M)
    g        = zeros(S,M)
    error_V = 1

    while error_V > eps_in

        for ii = 1:S
            for jj = 1:M
                U = reshape(U_matrix[ii,jj,:],1,M)
                U_v = U + beta*pi[ii,:]'*V
                TV[ii,jj], g[ii,jj] = findmax(U_v)
            end
        end

        # Compute approximation error with the sup norm
        error_V = maximum(maximum(abs.(TV - V)))
        V = deepcopy(TV)
        #println("error_V = ", error_V)
    end

    # Stationary distribution

    #   Rho is the transition matrix of the generalized (productivity and
    #   asset levels) state
    Rho = zeros(S*M,S*M)
    for i = 1:S
        for j = 1:M
            #Rho[(j - 1)*S + i, (g[i,j] - 1)*S + 1 : g[i,j]*S] = pi[i,:]'
            Rho[(j - 1)*S + i, (convert(Int64,g[i,j]) - 1)*S + 1 : convert(Int64,g[i,j])*S] = pi[i,:]
        end
    end

    # Find stationary distribution over the generalized state.
    Rho_sd = ones(1,S*M)/(S*M)
    error_Rho = 1

    while error_Rho > eps_in
        Rho_sd1 = Rho_sd*Rho
        error_Rho = maximum(abs.(Rho_sd1 - Rho_sd))
        Rho_sd = Rho_sd1
    end

    # Calculate aggregate asset supply
    g_vector = reshape(g,S*M,1)
    g_vec_int = convert(Array{Int64},g_vector)
    a_vector = a[g_vec_int]
    K = Rho_sd*a_vector

    K_out = K[1]

return K_out

end

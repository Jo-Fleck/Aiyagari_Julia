function My_Markovappr(lambda,sigma,m,N)

   #=
   This function approximates a first-order autoregressive process with a Markov chain
   y_t = lambda * y_(t-1) + u_t

   % u_t is a Gaussian white noise process with standard deviation sigma
   % m determines the width of discretized state space, Tauchen uses m=3
   % N is the number of possible states chosen to approximate
     the y_t process, usually N=9 should be fine

   % ymax=m*vary,ymin=-m*vary, ymax and ymin are two boundary points of
     the state space

   % Tran is the transition matrix of the Markov chain
   % probst is the invariant distribution matrix
   % s is the discretized state space of y_t

   =#

# FUN WILL NOW COMMENCE


# Discretize state space

stvy = sqrt(sigma^2/(1-lambda^2)) # standard deviation of y_t

ymax = m*stvy                     # upper boundary of state space
ymin = -ymax                      # lower boundary of state space

w = (ymax-ymin)/(N-1)             # length of interval

s = ymin:w:ymax                   # the discretized state space


# Calculate transition matrix
Tran = zeros(N, N)

for j = 1:N

   for k = 2:N-1

      Tran[j,k] = cdf(Normal(0,sigma),s[k]-lambda*s[j]+w/2) - cdf(Normal(0,sigma),s[k]-lambda*s[j]-w/2)

   end

   Tran[j,1] = cdf(Normal(0,sigma),s[1]-lambda*s[j]+w/2)
   Tran[j,N] = 1 - cdf(Normal(0,sigma),s[N]-lambda*s[j]-w/2)

end

# Test if rows do not add to one

if maximum( abs.( sum(Tran',1) - ones(1,N) )) > 1.0e-10

   str = find(Tran'.-ones(1,N))
   println("Error in transition matrix!")
   println("Row(s) not summing to one: ", str)

end


# Calculate the invariant distribution of the Markov Chain

Trans= Tran'
probst = (1/N)*ones(N,1) # initial distribution of states
test = 1;

  while test > 10.0^(-8)
      probst1 = Trans*probst
      test = maximum(abs.(probst1-probst))
      probst = probst1
   end

return Tran, s, probst

end

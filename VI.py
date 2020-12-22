
import numpy as np

#Variational inference algorithm for obtaining the parameters of the approximate posterior
def VI(X,iterations=10,E_t_guess=1,mu_0=0,lambda_0=1,a_0=3,b_0=1):
    N = len(X)
    X_sum = np.sum(X)
    X_square_sum = np.dot(X,X)

    mu_N = (lambda_0*mu_0 + X_sum)/(lambda_0+N)

    a_N = a_0+N/2

    #calc lambda_N with inital guess
    lambda_N = (lambda_0+N)*E_t_guess

    #dummy values to start loop
    b_N=np.inf
    b_N_old=-np.inf
    lambda_N_old=-np.inf

    i=0
    while i<iterations and (lambda_N-lambda_N_old)**2 + (b_N-b_N_old)**2 > 10**(-10):
        #calculate tao distrib
        E_mu = X_square_sum + lambda_0*(mu_0**2) - 2*(X_sum+lambda_0*mu_0)*mu_N + (N+lambda_0)*(1/(lambda_N**2)+mu_N**2)
        b_N_old = b_N
        b_N = b_0 + E_mu
        
        #calculate mu distrib
        E_tao = a_N/b_N
        lambda_N_old=lambda_N
        lambda_N= (lambda_0+N) * E_tao
        i+=1
        

    print(i, " iterations")
    return mu_N,lambda_N,a_N,b_N







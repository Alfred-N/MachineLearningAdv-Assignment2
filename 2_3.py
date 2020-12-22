import numpy as np
from plotGaussGam import *
from VI import VI
import matplotlib.pyplot as plt
from timeit import default_timer as time

N = 100
step=0.1
plotRangeMu=1
plotRangeTau=1.5
maxiterations = 1000

mu_0=0
lambda_0 = 1
a_0=3
b_0=1

np.random.seed(3)
X = generateGauss(N,0,1)

startt=time()
mu_N,lambda_N,a_N,b_N = VI(X,iterations=maxiterations,E_t_guess=1,mu_0=mu_0,lambda_0=lambda_0,a_0=a_0,b_0=b_0)
endt=time()
t = endt-startt
print("VI time =  ", t, "s")
print(mu_N,lambda_N,a_N,b_N)

plotGaussGam(plotRangeMu,plotRangeTau,step,X,truePosterior=True,mu_0=mu_0,lambda_0=lambda_0,a_0=a_0,b_0=b_0)
plotGaussGam(plotRangeMu,plotRangeTau,step,X,truePosterior=False,mu_0=mu_N,lambda_0=lambda_N,a_0=a_N,b_0=b_N)

plt.show()

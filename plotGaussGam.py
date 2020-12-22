import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
#Code for plotting Gaussian-Gamma distributions.


def generateGauss(N,mu,sig):
    X = np.random.normal(mu,sig,size=N)
    return X

#Calculates the values of the Gaussian-Gamma on specified grid, and plots a countour plot of the resulting distribution.
def plotGaussGam(plotRangeMu,plotRangeTau,step,X,truePosterior=True,mu_0=0,lambda_0=1,a_0=3,b_0=1):
    N = len(X)
    mu = np.arange(-plotRangeMu,plotRangeMu+step,step=step)
    tao= np.arange(0+step,plotRangeTau+step,step=step)
    MU,TAO = np.meshgrid(mu,tao)

    M= len(mu)
    plotRange=len(tao)
    posterior = np.zeros([plotRange,M])

    if truePosterior==True:
        #params of true posterior
        mu_N= (lambda_0*mu_0 + np.sum(X))/(N+lambda_0)
        a_N= a_0 + N/2
        b_N= b_0 + (1/2)*np.dot(X,X) + (lambda_0/2)*(mu_0**2) - ((N+lambda_0)/2)*(mu_N**2)


        gam=st.gengamma(a_N,b_N)
        for t_idx,t in enumerate(tao):
            for m_idx,m in enumerate(mu):
                gauss = st.norm(mu_N,(1/(t*(lambda_0+N))))
                gamFac=gam.pdf(t)
                gaussFac=gauss.pdf(m)
                posterior[t_idx,m_idx]= gamFac*gaussFac
        plt.xlabel("$\mu$",fontsize=18)
        plt.ylabel("$tau$",fontsize=18) 
        plt.contour(MU,TAO,posterior,colors="r",levels=4)
    else:
        gauss = st.norm(mu_0,1/lambda_0)
        gam=st.gengamma(a_0,b_0)
        for t_idx,t in enumerate(tao):
            for m_idx,m in enumerate(mu):
                gamFac=gam.pdf(t)
                gaussFac=gauss.pdf(m)
                posterior[t_idx,m_idx]= gamFac*gaussFac
        plt.xlabel("$\mu$")
        plt.ylabel("$tau$")  
        plt.contour(MU,TAO,posterior,colors="b",levels=4)
    return


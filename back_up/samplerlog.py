import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from argparse import RawTextHelpFormatter

#===============================================
# This is a test sampler for one single pixel, but
# structured such that it should be extendable
# to maps and additional constraints (priors)
# Content:
# class Parameter: parameter-specific information,
#   such as how to translate parameters to the model
#   (i.e. evaluate the model as function of the parameters)
#   and priors.
# class Data: here just an initialization of a test field.
#   Could contain a function to read in data. 
# methast: the actual sampler.
#===============================================
# class Parameter:
#   Container to evaluate the model as function of 
#   the parameters. Also provides (parameter-specific)
#   priors.
#-----------------------------------------------
class Parameter:
    def __init__(self):
        self.npar  = 2
        self.model = np.zeros(self.npar,dtype=object)

    # Evaluates model as function of parameters
    def EvalModel(self,thetapar):
        bx,by,bz      = self.Reconstruct(thetapar)
        cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
        q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
        u             = 2*bx*by/(bx**2+by**2) * cos2g
        phi           = 0.5*np.arctan2(u,q)
        pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach
        #pol           = bz/np.sqrt(bx**2+bz**2)
        self.model[0] = pol
        self.model[1] = phi

    # calculates the fiedl components as function of parameters (theta, phi)
    def Reconstruct(self,thetapar):
        # thetapar[0] = the
        # thetapar[1] = phi
        the  = thetapar[0]
        phi  = thetapar[1] 
        bx   = np.sin(the)*np.cos(phi)
        by   = np.sin(the)*np.sin(phi)
        bz   = np.cos(the)
        return bx,by,bz

    def GetModel(self,ind):
        return self.model[ind]

    # priors for parameters (theta, phi)
    def LogPrior(self,thetapar):
        lp = np.zeros(len(thetapar))
        if (thetapar[0] < 0.0 or thetapar[0] > np.pi):
            lp[0] = -100.0 
        if (thetapar[1] < 0.0 or thetapar[1] > np.pi):
            lp[1] = -100.0
        return np.sum(lp) 

#===================================
# class Data: container for initializing data set and calculating Likelihood.
#-----------------------------------
class Data:
    def __init__(self,bx0=1.0,by0=1.0,bz0=1.0):
        bx   = bx0
        by   = by0
        bz   = bz0
        btot = np.sqrt(bx**2+by**2+bz**2)
        bx   = bx/btot
        by   = by/btot
        bz   = bz/btot
        cos2g= (bx**2+by**2)/(bx**2+by**2+bz**2)
        q    = (by**2-bx**2)/(bx**2+by**2) * cos2g
        u    = 2*bx*by/(bx**2+by**2) * cos2g
        pol  = np.sqrt(u**2+q**2)
        #pol  = bz/np.sqrt(bx**2/bz**2)
        phi  = 0.5*np.arctan2(u,q)
        ph   = np.arccos(bx/np.sqrt((bx**2+by**2)))*180/np.pi
        th   = np.arccos(bz/np.sqrt(bx**2+by**2+bz**2))*180/np.pi
        print("[Data]: bx0=%13.5e by0=%13.5e bz0=%13.5e phi = %13.5e theta=%13.5e" % (bx,by,bz,ph,th))
        self.data = np.zeros(2,dtype=object)
        self.data[0] = pol
        self.data[1] = phi
        self.ndata   = 2
        self.bx0 = bx
        self.by0 = by
        self.bz0 = bz
        self.theta0 = th*np.pi/180.0
        self.phi0 = ph*np.pi/180.0

    def GetData(self,ind):
        return self.data[ind] 

    #plots map of likelihood determined over grid of phe and theta.
    def PlotLikelihood(self,par):
        nthe   = 100
        nphi   = 100
        minthe = 0.001
        maxthe = np.pi
        minphi = 0.0
        maxphi = np.pi
        the = np.linspace(minthe,maxthe,nthe)
        phi = np.linspace(minphi,maxphi,nphi)
        lik = np.zeros((nphi,nthe))
        for ithe in range(nthe):
            for iphi in range(nphi):
                thetapar       = np.array([the[ithe],phi[iphi]])
                par.EvalModel(thetapar)
                lik[iphi,ithe] = self.LogLikelihood(par,log=False)
                #print('phi, pol, lik = %13.5e %13.5e %13.5e' % (phi[iphi],the[ithe],lik[iphi,ithe]))
                
        fig1 = plt.figure(num=0,facecolor='white')
        ax0  = fig1.add_subplot(111)
        div  = make_axes_locatable(ax0)
        cax  = div.append_axes('right', size='5%', pad=0.05)
        img  = ax0.imshow(lik[::-1,:],cmap='viridis',extent=(minthe,maxthe,minphi,maxphi),aspect='auto')
        b,t  = ax0.get_ylim()
        l,r  = ax0.get_xlim()
        ax0.plot(np.ones(2)*self.theta0,[b,t],color='red')
        ax0.plot([l,r],np.ones(2)*self.phi0,color='red')
        ax0.set_ylabel(r'$\phi$')
        ax0.set_xlabel(r'$\theta$')
        fig1.colorbar(img,cax=cax,orientation='vertical')
        fig1.savefig('sampler_lik.png',format='png')
        plt.close(fig1)

    # logarithmic likelihood. Can do linear, for plotting purposes. May consider running everything linearly.
    def LogLikelihood(self,par,log=True):
        sq = 0.0
        for i in range(self.ndata):
            sq += (self.GetData(i)-par.GetModel(i))**2
        msq = -np.sum(sq)
        if not log:
            msq = np.exp(msq)
        return msq

# Generic sampler. See usage par.EvalModel and dat.LogLikelihood. 
def methast(R,theta0,delta,dat,par):
    theta      = np.zeros((par.npar,R))
    loglik     = np.zeros(R)
    theta[:,0] = theta0
    par.EvalModel(theta[:,0]) # evaluate model for initial condition
    loglik[0]  = dat.LogLikelihood(par) + par.LogPrior(theta[:,0]) # get likelihood (logarithm)
    lu         = np.log(np.random.rand(R))
    for r in range(1,R):
        while True: # need this to prevent issues with theta wrap-around
            theta[:,r] = theta[:,r-1] + delta*(2*np.random.rand(par.npar)-1.0)
            if (theta[0,r] >= 0.0 and theta[0,r] <= np.pi):
                break;
        theta[1,r] = theta[1,r] % (2.0*np.pi) # constrain phi between 0 and 2pi
        par.EvalModel(theta[:,r]) # evaluate the model given the new parameters
        loglik[r]  = dat.LogLikelihood(par) + par.LogPrior(theta[:,r])
        if lu[r] > loglik[r]-loglik[r-1]: # "negative" acceptance: since new state is already saved, overwrite if not accepted
            loglik[r] = loglik[r-1]
            theta[:,r]= theta[:,r-1] # restore old values
    return theta,loglik

#============================================
# restrict phi (P.A.) around observed P.A. (von Mises) and focus
# on recovering the inclination angle
def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-bx0",type=float,default=1.0,help='input field x')
    parser.add_argument("-by0",type=float,default=1.0,help='input field y')
    parser.add_argument("-bz0",type=float,default=1.0,help='input field z')
    parser.add_argument("-dtheta",type=float,default=0.2,help='stepsize for MH: theta')
    parser.add_argument("-dphi",type=float,default=0.2,help='stepsize for MH: phi')
    parser.add_argument("-bins",type=int,default=100,help='number of histogram bins')
    parser.add_argument("-R",type=int,default=100000,help='length of chain')
    args   = parser.parse_args()

    bx0 = args.bx0
    by0 = args.by0
    bz0 = args.bz0
    theta0 = np.array([45*np.pi/180.0,45*np.pi/180])
    delta  = np.array([args.dtheta,args.dphi])
    bins   = args.bins
    R      = args.R

    par = Parameter()
    dat = Data(bx0=bx0,by0=by0,bz0=bz0)

    theta,loglik = methast(R,theta0,delta,dat,par)
    bx,by,bz     = par.Reconstruct(theta)

    dat.PlotLikelihood(par)

    hist = np.zeros((par.npar,bins))
    x    = np.zeros((par.npar,bins))
    rang = np.arange(R)/R
    
    hist[0,:],e = np.histogram(theta[0,:],bins=bins,density=True)
    x[0,:]      = 0.5*(e[:-1]+e[1:])
    hist[1,:],e = np.histogram(theta[1,:],bins=bins,density=True)
    x[1,:]      = 0.5*(e[:-1]+e[1:])

    histxyz     = np.zeros((3,bins,2))
    histxyz[0,:,1],e = np.histogram(bx,bins=bins,density=True)
    histxyz[0,:,0]   = 0.5*(e[:-1]+e[1:]) 
    histxyz[1,:,1],e = np.histogram(by,bins=bins,density=True)
    histxyz[1,:,0]   = 0.5*(e[:-1]+e[1:])
    histxyz[2,:,1],e = np.histogram(bz,bins=bins,density=True)
    histxyz[2,:,0]   = 0.5*(e[:-1]+e[1:])

    fig0  = plt.figure(num=0,figsize=(10,6),facecolor='white')
    ax01  = fig0.add_subplot(2,2,1)
    ax01.plot(x[0,:],hist[0,:])
    b,t   = ax01.get_ylim()
    ax01.plot(np.ones(2)*dat.theta0,[b,t],color='grey')
    ax01.set_ylim([b,t])
    ax01.set_xlabel(r'$\theta$')
    ax02  = fig0.add_subplot(2,2,2)
    ax02.plot(x[1,:],hist[1,:])
    b,t   = ax02.get_ylim()
    ax02.plot(np.ones(2)*dat.phi0,[b,t],color='grey')
    ax02.set_ylim([b,t])
    ax02.set_xlabel(r'$\phi$')
    ax03  = fig0.add_subplot(2,2,3)
    ax03.plot(theta[0,:],rang)
    ax03.set_xlabel(r'$\theta$')
    ax03.set_ylabel('r')
    ax04  = fig0.add_subplot(2,2,4)
    ax04.plot(theta[1,:],rang)
    ax04.set_xlabel(r'$\phi$')
    ax04.set_ylabel('r')
    fig0.tight_layout()
    fig0.savefig('sampler_log.png',format='png')
    plt.show()


main()

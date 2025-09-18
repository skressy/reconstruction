import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from argparse import RawTextHelpFormatter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from scipy.integrate import quad

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

    # calculates the field components as function of parameters (theta, phi)
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
    def Prior(self,thetapar):
        p = np.ones(len(thetapar))
        if (thetapar[0] < 0.0 or thetapar[0] > np.pi):
            p[0] = 1e-60
        if (thetapar[1] < 0.0 or thetapar[1] > np.pi):
            p[1] = 1e-60
        return np.prod(p) 

#===================================
# class Data: container for initializing data set and calculating Likelihood.
#-----------------------------------
class Data:
    # def __init__(self,bx0=1.0,by0=1.0,bz0=1.0):
    def __init__(self,bx0,by0,bz0):
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
                lik[iphi,ithe] = self.Likelihood(par)
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
        plt.show()
        # plt.close(fig1)

    def Likelihood(self,par):
        sq = 0.0
        for i in range(self.ndata):
            sq += (self.GetData(i)-par.GetModel(i))**2
        msq = -np.sum(sq)
        return np.exp(msq)
    
# Exit container
# Generic sampler. See usage par.EvalModel and dat.LogLikelihood. 
def methast(R,parvec,delta,dat,par):
    theta      = np.zeros((par.npar,R)) # what is being returned
    lik        = np.zeros(R)            # what is being returned
    theta[:,0] = parvec                 # fill array with initial theta values
    par.EvalModel(theta[:,0]) # evaluate model for initial condition
    lik[0]     = dat.Likelihood(par) * par.Prior(theta[:,0]) # get likelihood (logarithm)
    u          = np.random.rand(R)
    for r in range(1,R):
        while True: # need this to prevent issues with theta wrap-around
            theta[:,r] = theta[:,r-1] + delta*(2*np.random.rand(par.npar)-1.0)
            if (theta[0,r] >= 0.0 and theta[0,r] <= np.pi):
                break;
        theta[1,r] = theta[1,r] % (2.0*np.pi) # constrain phi between 0 and 2pi
        par.EvalModel(theta[:,r]) # evaluate the model given the new parameters
        lik[r]     = dat.Likelihood(par) * par.Prior(theta[:,r])
        if u[r] > lik[r]/lik[r-1]: # "negative" acceptance: since new state is already saved, overwrite if not accepted
            lik[r]     = lik[r-1]
            theta[:,r] = theta[:,r-1] # restore old values
    return theta,lik

def peak(x):
    # Find peaks of a histogram, locates highest peak, returns indices
    peaks_inds, peaks = find_peaks(x, height=0)
    max_inds = np.argmax(peaks['peak_heights'])
    return peaks_inds, max_inds 

def bimodal(z,a_ratio,t1,s):
    return 1*np.exp(-(z-t1)**2/2/s**2)+a_ratio*np.exp(-(z-(np.pi-t1))**2/2/s**2)

def trapz(z,a_ratio,t1,s):
    h = (z[-1] - z[0]) / (len(z) - 1)
    f = bimodal(z,a_ratio,t1,s)
    I_trap = (h/2)*(f[0] + 2 * sum(f[1:len(z)-1]) + f[len(z)-1])
    return I_trap

def find_bimodal_fit(x, y, A_ratio, sig):
        peak_inds, peak_max_ind = peak(y)
        peak_ind = peak_inds[peak_max_ind]
        mu1     = x[peak_ind]
        expected=(A_ratio,mu1,sig)
        print('Initial values', expected)
        params,cov=curve_fit(bimodal,x,y,expected)
        eparams = np.sqrt(np.diag(cov))
        print('Fitted Values: ', 'A ratio',params[0],'theta',params[1],'sigma',params[2])
        # print('1 std error: ', np.sqrt(np.diag(cov)))
        bimodal_fit = bimodal(x,*params) 
        return bimodal_fit, params, eparams

def find_root_by_integration(x,a_ratio,t1,s):
    b0 = x[0]
    b1 = x[-1]
    # print('start and end', b0, b1)
    total_integral, _ = quad(bimodal, b0, b1, args=(a_ratio,t1,s))
    # print('total integral', total_integral)
    def cumulative_integral(x):
        integral_value, _ = quad(bimodal, b0,x,args=(a_ratio,t1,s))
        return integral_value - 0.5 * total_integral  # Want this to be 0
    solution = root_scalar(cumulative_integral, bracket=[b0,b1])
    if solution.converged:
        print(f"The value of x where the integral reaches 50% is: {solution.root}")
    else:
        print("Root finding did not converge.")
    return solution.root

#============================================
# restrict phi (P.A.) around observed P.A. (von Mises) and focus
# on recovering the inclination angle
def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-bx0",type=float,default=100,help='input field x')
    parser.add_argument("-by0",type=float,default=100,help='input field y')
    parser.add_argument("-bz0",type=float,default=0,help='input field z')
    parser.add_argument("-dtheta",type=float,default=0.2,help='stepsize for MH: theta')
    parser.add_argument("-dphi",type=float,default=0.2,help='stepsize for MH: phi')
    parser.add_argument("-bins",type=int,default=100,help='number of histogram bins')
    parser.add_argument("-R",type=int,default=8000,help='length of chain')
    parser.add_argument("-burnin",type=int,default=1000,help='length of burn in')
    args   = parser.parse_args()

    # grab starter values
    bx0 = args.bx0
    by0 = args.by0
    bz0 = args.bz0
    parvec = np.array([45*np.pi/180.0,45*np.pi/180])
    delta  = np.array([args.dtheta,args.dphi])
    bins   = args.bins
    R      = args.R
    burn   = args.burnin

    par = Parameter()
    dat = Data(bx0=bx0,by0=by0,bz0=bz0)
    thetaR = dat.theta0
    # dat.PlotLikelihood(par)    # plot 2d theta,phi grid of likelihood

    # set up peak dictionary:
    ntrials = 800 
    pdict = np.zeros((2,ntrials))

    for i in range(ntrials): 
        theta,lik = methast(R,parvec,delta,dat,par)
        bx,by,bz     = par.Reconstruct(theta)
        hist = np.zeros((par.npar,bins))
        x    = np.zeros((par.npar,bins))
        rang = np.arange(R)/R

        hist[0,:],e = np.histogram(theta[0,burn:],bins=bins,density=True) #burnin
        x[0,:]      = 0.5*(e[:-1]+e[1:])
        hist[1,:],e = np.histogram(theta[1,burn:],bins=bins,density=True) #burnin
        x[1,:]      = 0.5*(e[:-1]+e[1:])

        # find peaks of histogram:
        t_pinds,t_pmax = peak(hist[0,:])
        ph_pinds,ph_pmax = peak(hist[1,:])
        pdict[0][i] = x[0,:][t_pinds[t_pmax]]
        pdict[1][i] = x[1,:][ph_pinds[ph_pmax]]

    thist, e = np.histogram(pdict[0],bins=75,density=True)
    tx = 0.5*(e[:-1]+e[1:])
    phist, e = np.histogram(pdict[1],bins=bins,density=True)
    px = 0.5*(e[:-1]+e[1:])

    # need to normalize peaks histogram here
    thistnorm = thist/np.max(thist)
    phistnorm = phist/np.max(phist)

    # find max in histogram of peaks of theta,phi
    tpinds,tpmax = peak(thistnorm)
    ppinds,ppmax = peak(phistnorm)

    # find bimodal fit
    bifit, biparams, biparams_error = find_bimodal_fit(tx, thistnorm, A_ratio=0.75, sig=0.3)
    # find integrated median of theta
    theta_median = find_root_by_integration(tx,biparams[0],biparams[1],biparams[2])
    # calculate mmp
    sigma_dom = biparams[2]
    theta_peak = tx[tpinds[tpmax]]
    mmp = (theta_median - theta_peak)/sigma_dom
    print('MMP FOUND: ', mmp)

    # calculate shape parameter X = A1/A2*sign(theta1-theta2)
    sign = np.sign(biparams[1]-(np.pi-biparams[1]))
    Q = (1/biparams[0])*sign # A_ratio is A2/A1, so take reciprocal of it for A1/A2
    print('SHAPE PARAMETER:', Q)
    # if Q < 0, dominant peak on left
    # if Q > 0, dominant peak on right
    # magnitude is the fraction dominant peak is larger than smaller peak

    # set up plotting environment
    fig0  = plt.figure(num=0,figsize=(10,6),facecolor='white')
    # Plot Theta
    ax01  = fig0.add_subplot(3,2,1)
    ax01.plot(x[0,:],hist[0,:],linestyle='-',color='black')
    b,t   = ax01.get_ylim()
    ax01.set_ylim([b,t])
    ax01.set_xlabel(r'$\theta$')
    ax01.set_title(r'Theta')
    ax01.plot(np.ones(2)*dat.theta0,[b,t],color='grey') # plot original theta
    ax01.plot(np.ones(2)*x[0,:][t_pinds[t_pmax]],[b,t],color='red') # plot peak theta
    # Plot Phi
    ax02  = fig0.add_subplot(3,2,2)
    ax02.plot(x[1,:],hist[1,:], linestyle='-',color='black')
    b,t   = ax02.get_ylim()
    ax02.set_ylim([b,t])
    ax02.set_xlabel(r'$\phi$')
    ax02.plot(np.ones(2)*dat.phi0,[b,t],color='grey') # plot original phi
    ax02.plot(np.ones(2)*pdict[1][-1],[b,t],color='red') # plot peak phi

    # Plot Mixing Theta
    ax03  = fig0.add_subplot(3,2,3)
    ax03.plot(theta[0,:],rang)
    ax03.set_xlabel(r'$\theta$')
    ax03.set_ylabel('r')
    ax03.set_title(r'Mixing $\theta$')
    # Plot Mixing for Phi
    ax04  = fig0.add_subplot(3,2,4)
    ax04.plot(theta[1,:],rang)
    ax04.set_xlabel(r'$\phi$')
    ax04.set_ylabel('r')
    ax04.set_title(r'Mixing $\phi$')

    # Plot Histogram of samples of peaks
    # THETA
    ax05  = fig0.add_subplot(3,2,5)
    ax05.plot(tx,thistnorm)
    ax05.set_xlabel(r'$\theta$')
    ax05.set_title('Theta Peak Values')
    ax05.axvline(dat.theta0,color='black', linestyle='-',lw=3) # plot initial theta
    ax05.axvline(tx[tpinds[tpmax]],color='red', linestyle='-',lw=3) # plot peak theta
    ax05.axvline(theta_median,color='green', linestyle='-',lw=3) # plot median theta
    ax05.plot(tx,bifit,lw=3, alpha=1,label='model') # plot fit
    # PHI
    ax06  = fig0.add_subplot(3,2,6)
    ax06.plot(px,phistnorm)
    ax06.set_xlabel(r'$\Phi$')
    ax06.set_title('Phi Peak Values')
    ax06.axvline(dat.phi0,color='black',linestyle='-',lw=3) # plot original phi
    ax06.axvline(px[ppinds[ppmax]],color='red',linestyle='-',lw=3) # plot peak theta

    fig0.tight_layout() 
    plt.show()
    # # fig0.savefig('sampler_lin.png',format='png')





main()

# save every 50th trial:
# if i%100 == 0:
#     ax01.plot(x[0,:],hist[0,:],linestyle='-',color = 'gray')
#     ax02.plot(x[1,:],hist[1,:],linestyle='-',color='gray')


    # if i%100 == 0:
    #     ax01.plot(x[0,:],hist[0,:],linestyle='-',color = 'gray')
    #     ax02.plot(x[1,:],hist[1,:],linestyle='-',color='gray')

# calculate histograms of PEAK theta and phi
# e = x-values of bins, thist = frequency of bin, tx/px centers bins
# plt.plot(pdict[0],range(len(pdict[0])),marker='o',linestyle='none')
# plt.show()

# # Funtion fits bimodal curve to hist, finds median, peak, and sigma of dominant
# try:
#     bifit = find_bimodal_fit(tx, thist)
#     ax05.plot(tx,bifit,color='red',lw=3,label='model')
#     plt.show()
# except RuntimeError:
#     print('No convergence')
#     plt.show()


# calculate histograms of bx, by, bz
# histxyz     = np.zeros((3,bins,2))
# histxyz[0,:,1],e = np.histogram(bx,bins=bins,density=True)
# histxyz[0,:,0]   = 0.5*(e[:-1]+e[1:]) 
# histxyz[1,:,1],e = np.histogram(by,bins=bins,density=True)
# histxyz[1,:,0]   = 0.5*(e[:-1]+e[1:])
# histxyz[2,:,1],e = np.histogram(bz,bins=bins,density=True)
# histxyz[2,:,0]   = 0.5*(e[:-1]+e[1:])

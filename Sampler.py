import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import recon_funcs as rf
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from scipy.integrate import quad
from scipy.signal import find_peaks


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
#-----------------------------------------------
class Data: # class Data: container for initializing data set and calculating Likelihood.
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
        # print("[Data]: bx0=%13.5e by0=%13.5e bz0=%13.5e phi = %13.5e theta=%13.5e" % (bx,by,bz,ph,th))
        self.data = np.zeros(2,dtype=object)
        self.data[0] = pol
        self.data[1] = phi
        self.ndata   = 2
        self.bx0 = bx
        self.by0 = by
        self.bz0 = bz
        self.theta0 = th*np.pi/180.0
        self.phi0 = ph*np.pi/180.0

    def init_uq(self,u,q,cos2g):
        pol  = np.sqrt(u**2+q**2)
        phi  = 0.5*np.arctan2(u,q)

        self.data = np.zeros(2,dtype=object)
        self.data[0] = pol
        self.data[1] = phi
        self.ndata   = 2


    def GetData(self,ind):
        return self.data[ind] 

    #plots map of likelihood determined over grid of phi and theta.
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
#-----------------------------------------------
def methast(R,parvec,delta,dat,par): # Generic sampler. See usage par.EvalModel and dat.LogLikelihood. 
    # parvec[0] = theta
    # parvec[1] = phi
    theta      = np.zeros((par.npar,R)) # what is being returned
    lik        = np.zeros(R)            # what is being returned
    theta[:,0] = parvec                 # fill array with initial param values

    par.EvalModel(theta[:,0]) # evaluate model for initial condition -- this is where prolonged value goes
    lik[0]     = dat.Likelihood(par) * par.Prior(theta[:,0]) # get likelihood (logarithm)
    u          = np.random.rand(R)
    for r in range(1,R): # Only change and sample phi --> theta[1,r]
        # while True: # need this to prevent issues with theta wrap-around
        theta[1,r] = theta[1,r-1] + delta*(2*np.random.rand()-1.0) # delta step
        theta[0,r] = theta[0,r-1]
            # if (theta[0,r] >= 0.0 and theta[0,r] <= np.pi): # constrain theta between 0 and pi
            #     break;
        theta[1,r] = theta[1,r] % (2.0*np.pi) # constrain phi between 0 and 2pi
        par.EvalModel(theta[:,r]) # evaluate the model given the new parameters
        lik[r]     = dat.Likelihood(par) * par.Prior(theta[:,r])
        if u[r] > lik[r]/lik[r-1]: # "negative" acceptance: since new state is already saved, overwrite if not accepted
            lik[r]     = lik[r-1]
            theta[:,r] = theta[:,r-1] # restore old values
    return theta,lik



#============================================
# MCMC DRIVER ADAPTED
# restrict phi (P.A.) around observed P.A. (von Mises) and focus
# on recovering the inclination angle
# - - - - - - - - - - - - - - - - - - - - - 
# inputs: bx0,by0,bz0 are initial 2d maps of each value
#============================================
def mcmc_driver(data, prolonged, dtheta, dphi, rbins, R, nbins, ntrials, burn, plotting):

    # Reconstruct (restricted) data U12 to B12 (previously done outside of function)
    data3d = data # rf.init_recon_3D(data)
    # access each component from B12
    bx1 = data3d[0] # use as starting point
    by1 = data3d[1] # use as starting point
    bz1 = data3d[2] # this we sample -- where do we use this?
    # access guess or prolonged values from previous iteration
    px0 = prolonged[0] #ignore?
    py0 = prolonged[1] #ignore?
    pz0 = prolonged[2] # use as starting point
    # create arrays to store final sampled B values
    br0 = np.zeros(np.shape(bx1))
    br1 = np.zeros(np.shape(by1))
    br2 = np.zeros(np.shape(bz1))

    # for each cell in the 3D map
    for i in range(bx1.shape[0]):
        for j in range(by1.shape[0]):
            for k in range(bz1.shape[0]):
                # Initialize parameter functions
                par    = Parameter()
                # Initialize data; starting point
                dat    = Data(bx0=bx1[i][j][k],by0=by1[i][j][k],bz0=pz0[i][j][k]) # use prolonged data for bz, data for bx,by
                itheta = dat.theta0 # initialize theta
                iphi   = dat.phi0 # initialize phi, only sampling phi
                # Set initial conditions for theta and phi 
                parvec = np.array([itheta*np.pi/180.0,iphi*np.pi/180])
                delta  = np.array([dphi])

                # no looping through yet; just run one chain and plot mixing etc
                # Sample Phi
                theta,lik = methast(R,parvec,delta,dat,par)

                # generate historgram of phi values
                hist_phi, e = np.histogram(theta[1,burn:],bins=rbins,density=True) 
                x           = 0.5*(e[:-1]+e[1:])

                # find peak of histogram
                phi_inds, phi_max = rf.peak(hist_phi)
                phi_peak = x[phi_inds[phi_max]]
                # print('[checkpoint]: Phi Peak = ', phi_peak, 'Theta = ', itheta)

                bxr,byr,bzr = par.Reconstruct([itheta, phi_peak])

                br0[i][j][k] = bxr
                br1[i][j][k] = byr
                br2[i][j][k] = bzr
    
    if plotting == 1:
            
        # plot last, most recent run, histogram with peak
        plt.plot(x, hist_phi)
        plt.title('Phi')
        plt.show()

        # plot mixing
        plt.plot(range(len(theta[1,burn:])), theta[1,burn:])
        plt.title('sampler mixing')
        plt.ylabel('Phi')
        plt.show()

                ##############################
                # pdict = np.zeros((1,ntrials))

            # # for n in range(ntrials): 
            #     theta,lik = methast(R,parvec,delta,dat,par)
            #     hist = np.zeros((par.npar,rbins))
            #     x    = np.zeros((par.npar,rbins))

            #     hist[0,:],e = np.histogram(theta[0,burn:],bins=rbins,density=True) 
            #     x[0,:]      = 0.5*(e[:-1]+e[1:])
                # hist[1,:],e = np.histogram(theta[1,burn:],bins=rbins,density=True) 
                # x[1,:]      = 0.5*(e[:-1]+e[1:])

                # find peaks of histogram:
                # t_pinds,t_pmax = rf.peak(hist[0,:])
                # ph_pinds,ph_pmax = rf.peak(hist[1,:])
                # ph_pinds,ph_pmax = rf.peak(hist[0,:])
                # pdict[0][n] = x[0,:][t_pinds[t_pmax]]
                # pdict[0][n] = x[1,:][ph_pinds[ph_pmax]]

                # plt.plot(x[0,:],hist[0,:])

                # thist, e = np.histogram(pdict[0],bins=nbins,density=True)
                # tx = 0.5*(e[:-1]+e[1:])
                # phist, e = np.histogram(pdict[0],bins=nbins,density=True)
                # px = 0.5*(e[:-1]+e[1:])
                # normalize peaks histogram here - I dont think this does anything
                # thistnorm = thist/np.max(thist)
                # phistnorm = phist/np.max(phist)
                # find max in histogram of peaks of theta,phi
                # tpinds,tpmax = rf.peak(thistnorm)
                # ppinds,ppmax = rf.peak(phistnorm)
                # tpeak = tx[tpinds[tpmax]]
                # ppeak = px[ppinds[ppmax]]

                # reconstruct 3D field given theta and phi


    b22 = [br0,br1,br2]

    return b22

def simple_mcmc(u, dtheta, dphi, bins, R, burn):

    bx0 = u[0]
    by0 = u[1]
    bz0 = u[2]

    uc0 = np.zeros((bx0.shape[0],bx0.shape[1], bx0.shape[2]))
    uc1 = np.zeros((by0.shape[0],by0.shape[1], by0.shape[2]))
    uc2 = np.zeros((bz0.shape[0],bz0.shape[1], bz0.shape[2]))

    # for each cell in the 2D map
    for i in range(bx0.shape[0]):
        for j in range(bx0.shape[1]):
            for k in range(bx0.shape[2]):
                # Initialize theta and phi
                par = Parameter()
                dat = Data(bx0=bx0[i][j][k],by0=by0[i][j][k],bz0=bz0[i][j][k])
                itheta = dat.theta0
                iphi   = dat.phi0

                parvec = np.array([itheta*np.pi/180.0,iphi*np.pi/180])
                delta  = np.array([dtheta,dphi])

                theta,lik = methast(R,parvec,delta,dat,par)
                hist = np.zeros((par.npar,bins))
                x    = np.zeros((par.npar,bins))

                hist[0,:],e = np.histogram(theta[0,burn:],bins=bins,density=True) 
                x[0,:]      = 0.5*(e[:-1]+e[1:])
                hist[1,:],e = np.histogram(theta[1,burn:],bins=bins,density=True) 
                x[1,:]      = 0.5*(e[:-1]+e[1:])

                # find peaks of histogram:
                tpinds,tpmax = rf.peak(hist[0,:])
                ppinds,ppmax = rf.peak(hist[1,:])

                tpeak = x[0,:][tpinds[tpmax]]
                ppeak = x[1,:][ppinds[ppmax]]

                bxr,byr,bzr = par.Reconstruct([tpeak,ppeak])
                uc0[i][j][k] = bxr
                uc1[i][j][k] = byr
                uc2[i][j][k] = bzr

                # print(bx0[i][j][k]-bxr, by0[i][j][k]-byr, bz0[i][j][k]-bzr)

    uc = [uc0,uc1,uc2]

    return uc
# input: Bx,By,Bz
# convert: B components to theta and phi, spherical coords
# sample: theta, phi, compare to initial B components
# peak: find peak of theta, phi distribution
# hist of peak: find peak of peak (pdict)
# convert: theta and phi to B components
# return: best B components


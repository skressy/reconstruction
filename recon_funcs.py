import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from scipy.integrate import quad
from scipy.ndimage import convolve
import more_functions as mf
# from Reconstruction import multigrid

# a place to put all functions for reconstruction script (that for the most part are finished)

# -----------------------------------------------------------
# Find Peaks
# -----------------------------------------------------------
# Find peaks of a histogram, locates highest peak, returns indices
def peak(x):
    peaks_inds, peaks = find_peaks(x, height=0)     
    max_inds = np.argmax(peaks['peak_heights'])
    return peaks_inds, max_inds

# -----------------------------------------------------------
# Bimodal Function
# -----------------------------------------------------------
# returns bimodal function used for fitting
def bimodal(z,a_ratio,t1,s):
    return 1*np.exp(-(z-t1)**2/2/s**2)+a_ratio*np.exp(-(z-(np.pi-t1))**2/2/s**2)

# -----------------------------------------------------------
# Find Bimodal Fit
# -----------------------------------------------------------
# optimize a bimodal fit to distribution
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

# -----------------------------------------------------------
# Find Percentile Root via Integration
# -----------------------------------------------------------
# find the root via integration of a bimodal function, to some percentile
def find_root_by_integration(x,a_ratio,t1,s,percentile):
    b0 = x[0]
    b1 = x[-1]
    # print('start and end', b0, b1)
    total_integral, _ = quad(bimodal, b0, b1, args=(a_ratio,t1,s))
    # print('total integral', total_integral)
    def cumulative_integral(x):
        integral_value, _ = quad(bimodal, b0,x,args=(a_ratio,t1,s))
        return integral_value - percentile * total_integral  # Want this to be 0
    solution = root_scalar(cumulative_integral, bracket=[b0,b1])
    if solution.converged:
        print(f"The value of x where the integral reaches 50% is: {solution.root}")
    else:
        print("Root finding did not converge.")
    return solution.root

#============================================================
# Prolongation function 2.5D
#============================================================
def prolong_2D(u):
    J = u[0][0].shape[0]
    print('Prolonging... Level = ', J)

    # three arrays bx0,by0,bz0
    bx0 = np.zeros((J+2,J+2))
    by0 = np.zeros((J+2,J+2))
    bz0 = np.zeros((J+2,J+2))

    bx0[1:J+1,1:J+1] = u[0]
    by0[1:J+1,1:J+1] = u[1]
    bz0[1:J+1,1:J+1] = u[2]
    list_uc = [bx0,by0,bz0]

    i,j = np.meshgrid(np.arange(J,dtype=int),np.arange(J,dtype=int),indexing='xy') 
    uf = []
    for iuc in list_uc:
        mi0              = 0.5*(iuc[2:J+2,1:J+1]-iuc[0:J,1:J+1])
        mj0              = 0.5*(iuc[1:J+1,2:J+2]-iuc[1:J+1,0:J])
         
        uf0              = np.zeros((2*J,2*J))
        uf0[2*i  ,2*j  ] = u[0] - 0.25*mj0 - 0.25*mi0
        uf0[2*i  ,2*j+1] = u[0] + 0.25*mj0 - 0.25*mi0
        uf0[2*i+1,2*j  ] = u[0] - 0.25*mj0 + 0.25*mi0
        uf0[2*i+1,2*j+1] = u[0] + 0.25*mj0 + 0.25*mi0
        uf.append(uf0)
        
    return uf

#============================================================
# Prolongation function 3D
#============================================================
def prolong_3D(u):
    J = u[0][0].shape[0]
    print('Prolonging... Level = ', J)

    # three arrays bx0,by0,bz0
    bx0 = np.zeros((J+2,J+2,J+2))
    by0 = np.zeros((J+2,J+2,J+2))
    bz0 = np.zeros((J+2,J+2,J+2))

    bx0[1:J+1,1:J+1,1:J+1] = u[0]
    by0[1:J+1,1:J+1,1:J+1] = u[1]
    bz0[1:J+1,1:J+1,1:J+1] = u[2]
    list_uc = [bx0,by0,bz0]

    i,j,k = np.meshgrid(np.arange(J,dtype=int),np.arange(J,dtype=int),np.arange(J,dtype=int),indexing='xy') 
    uf = []
    for iuc in list_uc:
        mi0              = 0.5*(iuc[2:J+2,1:J+1,1:J+1]-iuc[0:J,1:J+1,1:J+1])
        mj0              = 0.5*(iuc[1:J+1,2:J+2,1:J+1]-iuc[1:J+1,0:J,1:J+1])
        mk0              = 0.5*(iuc[1:J+1,1:J+1,2:J+2]-iuc[1:J+1,1:J+1,0:J])

        uf0                    = np.zeros((2*J,2*J,2*J))
        uf0[2*i,2*j,2*k]       = u[0] - 0.25*mj0 - 0.25*mi0 - 0.25*mk0
        uf0[2*i,2*j+1,2*k]     = u[0] + 0.25*mj0 - 0.25*mi0 - 0.25*mk0
        uf0[2*i+1,2*j,2*k]     = u[0] - 0.25*mj0 + 0.25*mi0 - 0.25*mk0
        uf0[2*i+1,2*j+1,2*k]   = u[0] + 0.25*mj0 + 0.25*mi0 - 0.25*mk0
        uf0[2*i,2*j+1,2*k+1]   = u[0] + 0.25*mj0 - 0.25*mi0 + 0.25*mk0
        uf0[2*i,2*j,2*k+1]     = u[0] - 0.25*mj0 - 0.25*mi0 + 0.25*mk0
        uf0[2*i+1,2*j,2*k+1]   = u[0] - 0.25*mj0 + 0.25*mi0 + 0.25*mk0
        uf0[2*i+1,2*j+1,2*k+1] = u[0] + 0.25*mj0 + 0.25*mi0 + 0.25*mk0


        uf.append(uf0)
        
    return uf

#============================================================
# Restrict function
#============================================================
def restrict(u):
    J     = u[0][0].shape[0]
    JZ     = u[-1][0].shape[0]
    print('Restricting... Level = ', J)
    if np.shape(u[0]) != np.shape(u[-1]):
        i,j   = np.meshgrid(np.arange(J//2),np.arange(J//2),indexing='xy')
        k,l   = np.meshgrid(np.arange(JZ//2),np.arange(JZ//2),indexing='xy')
        uc0    = 0.25*(u[0][2*i,2*j]+u[0][2*i,2*j+1]+u[0][2*i+1,2*j]+u[0][2*i+1,2*j+1])
        uc1    = 0.25*(u[1][2*i,2*j]+u[1][2*i,2*j+1]+u[1][2*i+1,2*j]+u[1][2*i+1,2*j+1])
        uc2    = 0.25*(u[2][2*i,2*j]+u[2][2*i,2*j+1]+u[2][2*i+1,2*j]+u[2][2*i+1,2*j+1])

        uc3    = 0.25*(u[3][2*k,2*l]+u[3][2*k,2*l+1]+u[3][2*k+1,2*l]+u[3][2*k+1,2*l+1])
        uc     = [uc0,uc1,uc2,uc3]
    else:
        i,j   = np.meshgrid(np.arange(J//2),np.arange(J//2),indexing='xy')
        uc0    = 0.25*(u[0][2*i,2*j]+u[0][2*i,2*j+1]+u[0][2*i+1,2*j]+u[0][2*i+1,2*j+1])
        uc1    = 0.25*(u[1][2*i,2*j]+u[1][2*i,2*j+1]+u[1][2*i+1,2*j]+u[1][2*i+1,2*j+1])
        uc2    = 0.25*(u[2][2*i,2*j]+u[2][2*i,2*j+1]+u[2][2*i+1,2*j]+u[2][2*i+1,2*j+1])
        uc3    = 0.25*(u[3][2*i,2*j]+u[3][2*i,2*j+1]+u[3][2*i+1,2*j]+u[3][2*i+1,2*j+1])
        uc     = [uc0,uc1,uc2,uc3]

    return uc

def restrict_zeeman_min(u):
    J     = u[0].shape[0]
    JZ    = u[-1].shape[0]
    print('Restricting... Level = ', J)
    # print('Restricting... Level = ', JZ)
    if np.shape(u[0]) != np.shape(u[-1]):
        i,j   = np.meshgrid(np.arange(J//2),np.arange(J//2),indexing='xy')
        k,l   = np.meshgrid(np.arange(JZ//2),np.arange(JZ//2),indexing='xy')
        uc0    = 0.25*(u[0][2*i,2*j]+u[0][2*i,2*j+1]+u[0][2*i+1,2*j]+u[0][2*i+1,2*j+1])
        uc1    = 0.25*(u[1][2*i,2*j]+u[1][2*i,2*j+1]+u[1][2*i+1,2*j]+u[1][2*i+1,2*j+1])
        uc2    = 0.25*(u[2][2*i,2*j]+u[2][2*i,2*j+1]+u[2][2*i+1,2*j]+u[2][2*i+1,2*j+1])

        if JZ > 1:
            uc3    = 0.25*(u[3][2*k,2*l]+u[3][2*k,2*l+1]+u[3][2*k+1,2*l]+u[3][2*k+1,2*l+1])
        else:
            uc3 = u[3]
        
        uc     = [uc0,uc1,uc2,uc3]
    else:
        i,j   = np.meshgrid(np.arange(J//2),np.arange(J//2),indexing='xy')
        uc0    = 0.25*(u[0][2*i,2*j]+u[0][2*i,2*j+1]+u[0][2*i+1,2*j]+u[0][2*i+1,2*j+1])
        uc1    = 0.25*(u[1][2*i,2*j]+u[1][2*i,2*j+1]+u[1][2*i+1,2*j]+u[1][2*i+1,2*j+1])
        uc2    = 0.25*(u[2][2*i,2*j]+u[2][2*i,2*j+1]+u[2][2*i+1,2*j]+u[2][2*i+1,2*j+1])
        uc3    = 0.25*(u[3][2*i,2*j]+u[3][2*i,2*j+1]+u[3][2*i+1,2*j]+u[3][2*i+1,2*j+1])
        uc     = [uc0,uc1,uc2,uc3]

    return uc
#============================================================
# Initial Reconstruction 2.5D from Stokes U/Q ### DONT TOUCH!
#============================================================
def init_recon_2D(u): 
    U     = u[0]
    Q     = u[1]
    cos2g = u[2]

    bx = np.zeros((U.shape[0],U.shape[1]))
    by = np.zeros((U.shape[0],U.shape[1]))
    bz = np.zeros((U.shape[0],U.shape[1]))

    # for each cell in the 2D map
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            icos2g   = cos2g[i][j]
            iQ       = Q[i][j]
            iU       = U[i][j]

            iphi     = 0.5*np.arctan2(iU,iQ) # np.arctan takes radians
            isign    = np.sign(iphi)
            iby2     = 1/((np.tan(iphi))**2 + 1)
            ibx2     = 1 - iby2
            ibz2     = 1/icos2g * (ibx2 + iby2) * (1 - icos2g)

            # Deal with signage
            # BY
            iby = np.sqrt(iby2)
            if iphi > np.pi/2 and iphi < 3*np.pi/2:
                iby *= -1
            if iphi > -3*np.pi/2 and iphi < -np.pi/2:
                iby *= -1
            # BX
            ibx = np.sqrt(ibx2)
            if iphi > np.pi and iphi < 2*np.pi:
                ibx *= -1
            if iphi > -np.pi and iphi < 0:
                ibx *= -1

            # ibx      = np.sqrt(ibx2)
            # iby      = isign*np.sqrt(iby2)
            # iby      = np.sqrt(iby2)
            ibz      = np.sqrt(ibz2)
            # print(ibx,iby,ibz)
            #save; 
            bx[i][j] = ibx
            by[i][j] = iby
            bz[i][j] = ibz

    return [bx,by,bz]

#============================================================
# Initial Reconstruction 3D from Stokes U/Q
#============================================================
def init_recon_3D(u): 
    U     = u[0]
    Q     = u[1]
    cos2g = u[2]

    if U.shape[0] != U.shape[1]:
        print('Input field is not square')
        return

    bx = np.zeros((U.shape[0],U.shape[0],U.shape[0]))
    by = np.zeros((U.shape[0],U.shape[0],U.shape[0]))
    bz = np.zeros((U.shape[0],U.shape[0],U.shape[0]))

    # for each cell in the 2D map
    for n in range(bx.shape[0]):
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                icos2g   = cos2g[i][j]
                iQ       = Q[i][j]
                iU       = U[i][j]

                iphi     = 0.5*np.arctan2(iU,iQ) # np.arctan takes radians
                isign    = np.sign(iphi)
                iby2     = 1/((np.tan(iphi))**2 + 1)
                ibx2     = 1 - iby2
                ibz2     = 1/icos2g * (ibx2 + iby2) * (1 - icos2g)

                iby = np.sqrt(iby2)
                ibx = np.sqrt(ibx2)
                ibz = np.sqrt(ibz2)

                # BY
                if iphi > np.pi/2 and iphi < 3*np.pi/2:
                    iby *= -1
                if iphi > -3*np.pi/2 and iphi < -np.pi/2:
                    iby *= -1
                # BX
                if iphi > np.pi and iphi < 2*np.pi:
                    ibx *= -1
                if iphi > -np.pi and iphi < 0:
                    ibx *= -1
            
                #save
                bx[i][j][n] = ibx
                by[i][j][n] = iby
                bz[i][j][n] = ibz

    return [bx,by,bz]

#============================================================
# 2.5D Reconstruction from Stokes U/Q AND Zeeman
#============================================================
def zeeman_recon(u, plotting):
    # U, Q, COS2G, BLOS, plotting
    U     = u[0]
    Q     = u[1]
    COS2G = u[2]
    BLOS  = u[3]

    print(np.shape(U), np.shape(Q))

    nx, ny = U.shape[0], U.shape[0]

    # Step 1. Plane-of-sky orientation from Q,U
    phi = 0.5 * np.arctan2(U, Q)   # polarization angle

    # Step 2. Field magnitude from cos²γ and Zeeman
    Bmag = BLOS / np.sqrt(1 - COS2G)

    # Step 3. POS amplitude
    Bperp = np.sqrt(np.maximum(Bmag**2 - BLOS**2, 0))

    # Step 4. LOS component from Zeeman directly
    bz = BLOS.copy()
    
    # Step 5. Scale bx, by
    bx = np.sin(phi) * Bperp
    by = np.cos(phi) * Bperp

    # Total magnitude
    Btot = np.sqrt(bx**2 + by**2 + bz**2)

    # print(np.max(Btot))

    # Diagnostics / plotting
    if plotting == 1:
        fig, ax = plt.subplots(1, 5, figsize=(22,4))

        im0 = ax[0].imshow(bx, origin="lower", cmap="RdBu")
        ax[0].set_title("Bx map")
        plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(by, origin="lower", cmap="RdBu")
        ax[1].set_title("By map")
        plt.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(bz, origin="lower", cmap="RdBu")
        ax[2].set_title("Bz map (Zeeman)")
        plt.colorbar(im2, ax=ax[2],fraction=0.046, pad=0.04)

        im3 = ax[3].imshow(Bmag, origin="lower", cmap="viridis")
        ax[3].set_title("|B| total magnitude")
        plt.colorbar(im3, ax=ax[3],fraction=0.046, pad=0.04)

        # Quiver plot: POS field vectors
        step = 1   # downsample arrows
        ax[4].imshow(Bperp, origin="lower", cmap="viridis")
        i = np.arange(0, nx, step)
        j = np.arange(0, ny, step)
        ax[4].quiver(j, i, by[::step, ::step], bx[::step, ::step],
                    color="red", headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        ax[4].set_title("POS field (bx, by)")

        plt.tight_layout()
        plt.show()

        # step = 3
        # rf.visualize_3d(bx[::step, ::step], by[::step, ::step], bz[::step, ::step], name='Input Wavy Field',plotdex='ij')
        # plt.show()

        # rf.visualize_25d(bx[::step, ::step], by[::step, ::step], bz[::step, ::step], name='Input Wavy Field',plotdex='ij')
        # plt.show()
    
    return [bx,by,bz]

# #============================================================
# # OLD 2.5D Reconstruction from Stokes U/Q AND Zeeman
# #============================================================
# def zeeman_recon_min_2D(u, plotting):
#     # U, Q, COS2G, BLOS, plotting
#     U     = u[0]
#     Q     = u[1]
#     COS2G = u[2]
#     BLOS  = u[3]
#     nBLOS = np.zeros((U.shape[0],U.shape[0]))

#     print(np.shape(U), np.shape(Q))

#     nx, ny = U.shape[0], U.shape[1]

#     if U.shape[0] > 1:
#         for i in range(nx):
#             for j in range(ny):
#                 ifrac = i/nx
#                 jfrac = j/ny
#                 iblos = int(ifrac*np.shape(BLOS)[0])
#                 jblos = int(jfrac*np.shape(BLOS)[1])
#                 if BLOS.shape[0] > 1:
#                     nBLOS[i,j] = BLOS[iblos,jblos]
#                 else:
#                     nBLOS[i,j] = BLOS[0][0]
    
#     else:
#         nBLOS = BLOS
    
#     print('U', np.shape(U), 'BLOS:', np.shape(nBLOS))

#     # Step 1. Plane-of-sky orientation from Q,U
#     phi = 0.5 * np.arctan2(U, Q)   # polarization angle

#     # Step 2. Field magnitude from cos²γ and Zeeman
#     Bmag = nBLOS / np.sqrt(1 - COS2G)

#     # Step 3. POS amplitude
#     Bperp = np.sqrt(np.maximum(Bmag**2 - nBLOS**2, 0))

#     # Step 4. LOS component from Zeeman directly
#     bz = nBLOS.copy()
    
#     # Step 5. Scale bx, by
#     bx = np.sin(phi) * Bperp
#     by = np.cos(phi) * Bperp

#     # Total magnitude
#     Btot = np.sqrt(bx**2 + by**2 + bz**2)

#     # print(np.max(Btot))

#     # Diagnostics / plotting
#     if plotting == 1:
#         fig, ax = plt.subplots(1, 5, figsize=(22,4))

#         im0 = ax[0].imshow(bx, origin="lower", cmap="RdBu")
#         ax[0].set_title("Bx map")
#         plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

#         im1 = ax[1].imshow(by, origin="lower", cmap="RdBu")
#         ax[1].set_title("By map")
#         plt.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)

#         im2 = ax[2].imshow(bz, origin="lower", cmap="RdBu")
#         ax[2].set_title("Bz map (Zeeman)")
#         plt.colorbar(im2, ax=ax[2],fraction=0.046, pad=0.04)

#         im3 = ax[3].imshow(Bmag, origin="lower", cmap="viridis")
#         ax[3].set_title("|B| total magnitude")
#         plt.colorbar(im3, ax=ax[3],fraction=0.046, pad=0.04)

#         # Quiver plot: POS field vectors
#         step = 1   # downsample arrows
#         ax[4].imshow(Bperp, origin="lower", cmap="viridis")
#         i = np.arange(0, nx, step)
#         j = np.arange(0, ny, step)
#         ax[4].quiver(j, i, by[::step, ::step],bx[::step, ::step],
#                     color="red", headaxislength=0, headlength=0, headwidth=1, pivot='middle')
#         ax[4].set_title("POS field (bx, by)")

#         plt.tight_layout()
#         plt.show()

#         # step = 3
#         # rf.visualize_3d(bx[::step, ::step], by[::step, ::step], bz[::step, ::step], name='Input Wavy Field',plotdex='ij')
#         # plt.show()

#         # rf.visualize_25d(bx[::step, ::step], by[::step, ::step], bz[::step, ::step], name='Input Wavy Field',plotdex='ij')
#         # plt.show()
    
#     return [bx,by,bz]

#============================================================
# 2.5D Reconstruction from Stokes U/Q AND Zeeman
#============================================================
def zeeman_recon_min_2D(u, plotting, near_neighbor):
    # U, Q, COS2G, BLOS, plotting
    U     = u[0]
    Q     = u[1]
    COS2G = u[2]
    BLOS  = u[3]

    plt.imshow(BLOS, origin="lower", cmap="viridis")
    print('num. of BLOS values', len(np.where(BLOS != -1)))
    print('SHAPE BLOS', np.shape(BLOS), 'SHAPE U', np.shape(U))

    # First, check to see if we have any BLOS values at all
    num_BLOS_values = len(np.where(BLOS != -1)[0])
    if num_BLOS_values == 0:
        print('No BLOS values found, performing pure polarimetric reconstruction')
        bx, by, bz = mf.precon(U, Q, COS2G)
    
    else:
        print('BLOS values found, performing Zeeman + polarimetric reconstruction')

        # if near_neighbor is True, we will fill in missing BLOS values with nearest neighbor
        if near_neighbor == True:
            nx, ny = U.shape[0], U.shape[1]
            new_BLOS = np.zeros((nx, ny))
            # For levels larger than 1 cell
            if BLOS.shape[0] > 1:
                for i in range(nx):
                    for j in range(ny):
                        # For each cell, find nearest BLOS value that isn't -1, assign to new_BLOS array
                        # so that each cell has some BLOS value. 
                        index = mf.nearest_non_minus_one(BLOS, (i,j), allow_diagonals=False)
                        new_BLOS[i,j] = BLOS[index]
            # 1 cell lvl
            else:
                new_BLOS = BLOS
            # Reconstruct Bx, By, Bz with new BLOS array
            bx, by, bz = mf.zrecon(U, Q, new_BLOS, COS2G)
        
        # If near_neighbor is False, only reconstruct where we have BLOS values. Otherwise, use polarimetric only.
        if near_neighbor == False:
            
            nx, ny = U.shape[0], U.shape[1]
            bx = np.zeros((nx,ny))
            by = np.zeros((nx,ny))
            bz = np.zeros((nx,ny))
            # For levels larger than 1 cell
            if BLOS.shape[0] > 1:
                # have to step thru each cell, and reconstruct based on BLOS information available
                for i in range(nx):
                    for j in range(ny):
                        if BLOS[i][j] != -1:
                            BX, BY, BZ = mf.zrecon(U[i][j], Q[i][j], BLOS[i][j], COS2G[i][j])
                            bx[i][j] = BX
                            by[i][j] = BY
                            bz[i][j] = BZ
                        else:
                            BX, BY, BZ = mf.precon(U[i][j], Q[i][j], COS2G[i][j])
                            bx[i][j] = BX
                            by[i][j] = BY
                            bz[i][j] = BZ

    # Diagnostics / plotting
    if plotting == 1:
        fig, ax = plt.subplots(1, 4, figsize=(22,4))

        im0 = ax[0].imshow(bx, origin="lower", cmap="RdBu")
        ax[0].set_title("Bx map")
        plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(by, origin="lower", cmap="RdBu")
        ax[1].set_title("By map")
        plt.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(bz, origin="lower", cmap="RdBu")
        ax[2].set_title("Bz map (Zeeman)")
        plt.colorbar(im2, ax=ax[2],fraction=0.046, pad=0.04)

        # Quiver plot: POS field vectors
        step = 1   # downsample arrows
        ax[3].imshow(bz, origin="lower", cmap="viridis")
        i = np.arange(0, nx, step)
        j = np.arange(0, ny, step)
        ax[3].quiver(j, i, by[::step, ::step],bx[::step, ::step],
                    color="red", headaxislength=0, headlength=0, headwidth=1, pivot='middle')
        ax[3].set_title("POS field (bx, by)")

        plt.tight_layout()
        plt.show()
    
    return [bx,by,bz]

#######################################
def zeeman_recon_min_3D(u): 
    # This code is not generic and can't work with 2.5D Zeeman reconstruction, need to incorporate both at some point.
    U     = u[0]
    Q     = u[1]
    COS2G = u[2]
    BLOS  = u[3]
    nBLOS = np.zeros((U.shape[0],U.shape[0]))
    print('new blos array size', np.shape(nBLOS))

    # need to grab closest BLOS value, not necessarily have information in each cell.
    # calculates fraction of index of i,j and then populates the new BLOS array with 
    # respective information. This way, BLOS array is the same size as U/Q/COS2G, and
    # BLOS information is extrapolated and smoothed. 
    if U.shape[0] > 1:
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                ifrac = i/np.shape(U)[0]
                jfrac = j/np.shape(U)[1]
                iblos = int(ifrac*np.shape(BLOS)[0])
                jblos = int(jfrac*np.shape(BLOS)[1])
                if BLOS.shape[0] > 1:
                    nBLOS[i,j] = BLOS[iblos,jblos]
                else:
                    # print('BLOS value',BLOS[0][0])
                    nBLOS[i,j] = BLOS[0][0]
        # print('BLOS size', BLOS.shape[0])
    
    else:
        nBLOS = BLOS
    
    print('U', np.shape(U), 'BLOS:', np.shape(nBLOS))

    if U.shape[0] != U.shape[1]:
        print('Input field is not square')
        return

    bx = np.zeros((U.shape[0],U.shape[0],U.shape[0]))
    by = np.zeros((U.shape[0],U.shape[0],U.shape[0]))
    bz = np.zeros((U.shape[0],U.shape[0],U.shape[0]))

    # for each cell in the 2D map
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            for n in range(bx.shape[0]):

                iQ        = Q[i][j]
                iU        = U[i][j]
                iCOS2G    = COS2G[i][j]
                iBLOS     = nBLOS[i][j]

                # Step 1. Plane-of-sky orientation from Q,U
                iphi = 0.5 * np.arctan2(iU, iQ)   # polarization angle

                # Step 2. Field magnitude from cos²γ and Zeeman
                Bmag = iBLOS / np.sqrt(1 - iCOS2G)

                # Step 3. POS amplitude
                Bperp = np.sqrt(np.maximum(Bmag**2 - iBLOS**2, 0))

                # Step 4. LOS component from Zeeman directly
                ibz = iBLOS.copy()
                
                # Step 5. Scale bx, by
                ibx = np.sin(iphi) * Bperp
                iby = np.cos(iphi) * Bperp

                # Total magnitude
                Btot = np.sqrt(bx**2 + by**2 + bz**2)
                #save
                # print('iPHI = ', iphi)

                # print('iBX = ', ibx)
    
                bx[i][j][n] = ibx
                by[i][j][n] = iby
                bz[i][j][n] = ibz
    
    # print('BTOT:', np.shape(Btot))

    return [bx,by,bz]

#============================================================
# Reconstruct 2.5D U,Q from Bx,By,Bz #### DONT TOUCH!
#============================================================
def UQ_reconstruct_2D(bx,by,bz):
    u     = np.zeros((bx.shape[0],bx.shape[1]))
    q     = np.zeros((bx.shape[0],bx.shape[1]))
    cos2g = np.zeros((bx.shape[0],bx.shape[1]))
    phi   = np.zeros((bx.shape[0],bx.shape[1]))
    pol   = np.zeros((bx.shape[0],bx.shape[1]))

    # for each cell in the 2D map
    for i in range(bx.shape[0]):
        for j in range(bx.shape[1]):
            ibx     = bx[i][j]
            iby     = by[i][j]
            ibz     = bz[i][j]

            icos2g  = (ibx**2+iby**2)/(ibx**2+iby**2+ibz**2)
            iq      = (iby**2-ibx**2)/(ibx**2+iby**2) * icos2g
            iu      = 2*ibx*iby/(ibx**2+iby**2) * icos2g
            iphi    = 0.5*np.arctan2(iu,iq)
            ipol    = np.sqrt(iu**2+iq**2)

            u[i][j]     = iu
            q[i][j]     = iq
            cos2g[i][j] = icos2g
            phi[i][j]   = iphi
            pol[i][j]   = ipol

    return u,q,cos2g,phi,pol

#============================================================
# Reconstruct 3D U,Q from Bx,By,Bz 
# For now, I can just take the mean of each value in the z 
# direction because z isn't varying, but at some point, 
# I'll need to integrate along the LOS like we do in originally 
# calculating the stokes u/Q in Athena++
#============================================================
def UQ_reconstruct_3D(bx,by,bz):
    u     = np.zeros((bx.shape[0],bx.shape[1]))
    q     = np.zeros((bx.shape[0],bx.shape[1]))
    cos2g = np.zeros((bx.shape[0],bx.shape[1]))
    phi   = np.zeros((bx.shape[0],bx.shape[1]))
    pol   = np.zeros((bx.shape[0],bx.shape[1]))

    # for each cell in the 2D map
    for i in range(bx.shape[0]):
        for j in range(bx.shape[1]):
            ibx     = bx[i][j]
            iby     = by[i][j]
            ibz     = bz[i][j]

            icos2g  = (ibx**2+iby**2)/(ibx**2+iby**2+ibz**2)
            iq      = (iby**2-ibx**2)/(ibx**2+iby**2) * icos2g
            iu      = 2*ibx*iby/(ibx**2+iby**2) * icos2g
            iphi    = 0.5*np.arctan2(iu,iq)
            # iphi    = np.arctan(ibx,iby)
            ipol    = np.sqrt(iu**2+iq**2)

            u[i][j]     = np.mean(iu)
            q[i][j]     = np.mean(iq)
            cos2g[i][j] = np.mean(icos2g)
            phi[i][j]   = np.mean(iphi)
            pol[i][j]   = np.mean(ipol)

    return u,q,cos2g,phi,pol

#============================================================
# Laplacian convolution function for a 2D array
#============================================================
def convolve_laplacian(array):
    # Define a 2D Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    # Convolve the 2D array with the Laplacian kernel
    # The mode='reflect' parameter handles the array boundaries by reflecting the array at the borders.
    convolved_array = convolve(array, laplacian_kernel, mode='reflect')

    return convolved_array

#============================================================
# Regularization function ### DON'T TOUCH!
#============================================================
def regularization(Uin,Uout,weight): # make so it can take in arrays
    # r = result of laplacian convolution of cell
    # R = sum square of r
    # weight = a regularization parameter controlling the weight of the coherence term
    # Q = sum square of residual; cost function
    # J = final regularization term
    print('Regularizing...')
    r = convolve_laplacian(Uout)
    Q = np.sqrt(np.sum((Uin[0] - Uout[0])**2 + (Uin[1] - Uout[1])**2 + (Uin[2] - Uout[2])**2))
    R = np.sum(r^2)
    J = weight*R + Q
    return J

#============================================================
# Regularization Function (TEMPORARY)
# THESE CAN BE DELETED ONCE THE FINAL REGULARIZATION FUNCTION IS WORKING
#============================================================
def regulate(data, guess):
    # This needs to apply spatial coherence on Bx,By,Bz arrays, 
    # find the cost function and apply the regularization term.

    # I think we then need to add a way of minimizing the cost function/regularization.

    print('Regulating...')
    return guess

#============================================================
# Calculate Residual Cycle
#============================================================
def residual(U_i,Q_i,U_r,Q_r):
    # calculate residual between U,Q in and U,Q out
    resy = np.sqrt(np.mean((U_i - U_r)**2 + (Q_i - Q_r)**2))
    print(resy)
    return resy

# -----------------------------------------------------------
# -----------------------------------------------------------
# Plotting functions below
# ----------------------------------------------------------
# -----------------------------------------------------------
# plot reconstructed field in 3D
# -----------------------------------------------------------
def visualize_3d(bx, by, bz, name, plotdex):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.meshgrid(np.arange(bz.shape[0]),
                          np.arange(bz.shape[0]),
                          np.arange(bz.shape[0]),indexing=plotdex)
    ax.quiver(X, Y, Z, bx, by, bz, length=0.5, normalize=True, 
            arrow_length_ratio=0, color='dodgerblue')
    ax.set_xlabel('Bx')
    ax.set_ylabel('By')
    ax.set_zlabel('Bz')
    ax.set_title(name)
    plt.show()

# -----------------------------------------------------------
# plot generated field in 3D
# -----------------------------------------------------------
def visualize_gen_3D(X,Y,Z,bx,by,bz):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, bx, by, bz, length=0.05, normalize=True,linewidth=1.5, arrow_length_ratio=0, color='dodgerblue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Magnetic Field: Hourglass Configuration')
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# GET THE FUCKING INDICES RIGHT BOY ##### DONT TOUCH!
# -----------------------------------------------------------
def visualize_25d(bx, by, bz, name, plotdex):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.meshgrid(np.arange(bx.shape[0]), 
                          np.arange(by.shape[0]), 
                          np.arange(bz.shape[0]), indexing=plotdex)
    
    ax.quiver(X,Y,Z,bx,by,bz,length=0.5, normalize=True, arrow_length_ratio=0, pivot='middle', color='dimgray')
    # for i in range(X.shape[0]):
    #     for j in range(Y.shape[0]):
    #         ax.quiver(X[i][j], Y[i][j], Z[i][j], bx[i][j], by[i][j], bz[i][j], 
    #                   length=0.5, normalize=True, 
    #                   arrow_length_ratio=0, pivot='middle', color='dimgray')
    ax.set_xlabel('Bx')
    ax.set_ylabel('By')
    ax.set_zlabel('Bz')
    ax.set_title(name)
    plt.show()


# -----------------------------------------------------------
# plot U/Q in 2D, functions for both mesh and xy coords #### DONT TOUCH!
# -----------------------------------------------------------
def visual_UQ(U,Q,label,plotdex):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.set_title(label)
    X, Y = np.meshgrid(np.arange(U.shape[0]), np.arange(Q.shape[0]),indexing=plotdex)
    phi = 0.5*np.arctan2(U,Q)
    x = np.cos(phi)
    y = np.sin(phi)
    ax.quiver(X, Y, x, y, headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

# -----------------------------------------------------------
# plot U/Q in 2D with depolarization
# -----------------------------------------------------------
def visual_UQ_depol(U,Q,pol,label,plotdex):

    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]), indexing=plotdex)
    factor = 1 / np.max(pol)

    phi = 0.5 * np.arctan2(U, Q)
    x_scaled = np.sin(phi) * pol * factor
    y_scaled = np.cos(phi) * pol * factor

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.set_title(label)
    # Plot scaled vectors in blue
    ax.quiver(Y,X,y_scaled,x_scaled, scale=2, scale_units='xy', headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue', alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


# -----------------------------------------------------------
# plot slice of 3D field ### DONT TOUCH!
# -----------------------------------------------------------
def visualize_slice(x, y, label, plotdex):
    X, Y = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[0]), indexing=plotdex)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    ax.quiver(X, Y, x, y, headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='dimgray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Slice of Reconstructed '+ label)
    plt.axis('equal')
    plt.show()

# -----------------------------------------------------------
# plot vecors in 2D
# -----------------------------------------------------------
def plot_vector(angle_rad, plt_ax, magnitude):
    x = magnitude * np.cos(angle_rad)
    y = magnitude * np.sin(angle_rad)

    plt_ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1)


# -----------------------------------------------------------
# plot Hourglass B field components for 3 axis
# -----------------------------------------------------------
def plot_slice_3views(X,Y,Z,bx,by,bz):
        fig, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].set_title('XY-Plane (Z ≈ 0)')
        axs[0].quiver(X[:, :, 0], Y[:, :, 0], bx[:, :, 0], by[:, :, 0],
                    headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='dodgerblue')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].axis('equal')

        axs[1].set_title('ZX-Plane (Y ≈ 0)')
        axs[1].quiver(Z[0,:,:], X[0,:,:], bz[0,:,:], bx[0,:,:],
                headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='dodgerblue')
        axs[1].set_xlabel('Z')
        axs[1].set_ylabel('X')
        axs[1].axis('equal')

        axs[2].set_title('YZ-Plane (X ≈ 0)')
        axs[2].quiver(Y[:,0,:], Z[:,0,:], by[:,0,:], bz[:,0,:],
                headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='dodgerblue')
        axs[2].set_xlabel('Y')
        axs[2].set_ylabel('Z')
        axs[2].axis('equal')

        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------
# plot PAs in 2D; NEEDS UPDATING
# -----------------------------------------------------------
def plot_PAs(PAs):
    plt.figure(figsize=(7,7))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position Angles')
    for i in range(len(PAs[0])):
        for j in range(len(PAs[1])):
            iphi = PAs[i][j]
            x = 1 * np.cos(iphi)
            y = 1 * np.sin(iphi)
            plt.quiver(i, j, x, y, angles='xy', scale_units='xy', scale=1, headlength=0, headaxislength=0, headwidth=0)
    plt.show()

# relayer data
def relayer_data(U,Q,COS2G):
    newU = np.zeros(U.shape)
    newQ = np.zeros(Q.shape)
    newCOS = np.zeros(COS2G.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            for k in range(U.shape[2]):
                newU[k][i][j] = U[i][j][k]
                newQ[k][i][j] = Q[i][j][k]
                newCOS[k][i][j] = COS2G[i][j][k]
    return newU, newQ, newCOS
# -----------------------------------------------------------
# Wavy Field; Loop thru Bz
# STILL NEEDS TO BE TESTED
# NEED TO FIX MULTIGRID ISSUE
# -----------------------------------------------------------
# def wave_field_loop(nloops):

#     Bzlist    = np.linspace(-1,1,nloops)
#     residuals = np.zeros(len(Bzlist))
#     PAi_array = np.zeros(len(Bzlist))
#     PAr_array = np.zeros(len(Bzlist))

#     for i, iBz in enumerate(Bzlist):
#         UQpol_array               = dg.wave_3d(amplitude=1, frequency=1, box_length=16, z_contribution=iBz)
#         d3_field                  = multigrid(UQpol_array)
#         Ur, Qr, cos2gr, PAr, polr = UQ_reconstruct_2D(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][1])
#         res                       = residual(UQpol_array[0], UQpol_array[1], Ur, Qr)

#         PAi     = np.mean(0.5*np.arctan2(UQpol_array[0],UQpol_array[1]))
#         PAi_map = 0.5*np.arctan2(UQpol_array[0],UQpol_array[1])

#         residuals[i] = res
#         PAi_array[i] = PAi*(180/np.pi)
#         PAr_array[i] = np.mean(PAr*(180/np.pi))


#     # plotting
#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     ax[0].plot(Bzlist, residuals, marker='o', linestyle='none')
#     ax[0].set_xlabel('Bz value')
#     ax[0].set_ylabel('Residual')

#     im1 = ax[1].imshow(PAi_map, origin='lower', aspect='auto', cmap='viridis')
#     cbar1 = fig.colorbar(im1, ax=ax[1])
#     ax[1].set_title('Initial PA')
#     ax[1].set_xlabel('X')
#     ax[1].set_ylabel('Y')

#     im2 = ax[2].imshow(PAr, origin='lower', aspect='auto', cmap='viridis')
#     cbar2 = fig.colorbar(im2, ax=ax[2])
#     ax[2].set_title('Reconstructed PA')
#     ax[2].set_xlabel('X')
#     ax[2].set_ylabel('Y')

#     ax[1].set_xlabel('U Value')
#     ax[1].set_ylabel('Q Value', labelpad=0.01)
#     ax[1].grid()
#     cbar1.set_label(label='PAr', labelpad=0.01)
#     plt.show()

# -----------------------------------------------------------
# Uniform Field Loop; Loop through U and Q
# STILL NEEDS TO BE TESTED
# NEED TO FIX MULTIGRID ISSUE
# -----------------------------------------------------------
# def uniform_field_loop(nU, nQ):
#     Ulist = np.linspace(-1,1,nU)
#     Qlist = np.linspace(-1,1,nQ)
#     residuals = np.zeros((len(Ulist), len(Qlist)))
#     PAi_array = np.zeros((len(Ulist), len(Qlist)))
#     PAr_array = np.zeros((len(Ulist), len(Qlist)))

#     for i, U in enumerate(Ulist):
#         for j, Q in enumerate(Qlist):
#             UQpol_array               = dg.uqpol_generator(U0=U,Q0=Q,box_length=8)
#             d3_field                  = multigrid(UQpol_array)
#             Ur, Qr, cos2gr, PAr, polr = UQ_reconstruct_2D(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][2])
#             res                       = residual(UQpol_array[0], UQpol_array[1], Ur, Qr)
#             PAi                       = np.mean(0.5*np.arctan2(UQpol_array[0],UQpol_array[1]))

#             residuals[i][j] = res
#             PAi_array[i][j] = PAi*(180/np.pi)
#             PAr_array[i][j] = np.mean(PAr*(180/np.pi))

#     fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

#     im0 = ax[0].imshow(PAi_array, extent=[Ulist.min(), Ulist.max(), Qlist.min(), Qlist.max()], origin='lower', aspect='auto', cmap='viridis')
#     ax[0].set_xlabel('U Value')
#     ax[0].set_ylabel('Q Value', labelpad=0.01)
#     ax[0].grid()
#     cbar0 = fig.colorbar(im0, ax=ax[0])
#     cbar0.set_label(label='PAi', labelpad=0.01)

#     im1 = ax[1].imshow(PAr_array, extent=[Ulist.min(), Ulist.max(), Qlist.min(), Qlist.max()], origin='lower', aspect='auto', cmap='viridis')
#     ax[1].set_xlabel('U Value')
#     ax[1].set_ylabel('Q Value', labelpad=0.01)
#     ax[1].grid()
#     cbar1 = fig.colorbar(im1, ax=ax[1])
#     cbar1.set_label(label='PAr', labelpad=0.01)

#     im2 = ax[2].imshow(residuals, extent=[Ulist.min(), Ulist.max(), Qlist.min(), Qlist.max()], origin='lower', aspect='auto', cmap='viridis')
#     ax[2].set_xlabel('U Value')
#     ax[2].set_ylabel('Q Value', labelpad=0.01)
#     ax[2].axhline(y=0, color='white', linestyle='-')
#     ax[2].axvline(x=0, color='white', linestyle='-')
#     ax[2].grid()
#     cbar2 = fig.colorbar(im2, ax=ax[2])
#     cbar2.set_label(label='Residual', labelpad=0.01)

#     sc = ax[3].scatter(PAi_array, PAr_array, c=residuals, cmap='viridis')
#     ax[3].set_xlabel('PAi Value')
#     ax[3].set_ylabel('PAr Value', labelpad=0.01)
#     ax[3].grid()
#     cbar3 = fig.colorbar(sc, ax=ax[3])
#     cbar3.set_label(label='Residual', labelpad=0.01)

#     plt.tight_layout()
#     plt.show()



       


    # # plot image array
    # fig, axs = plt.subplots(1,3,figsize=(14,4))
    # phi_im = axs[2].imshow(phi[:,2,:], origin='lower',cmap='viridis',aspect='auto')
    # fig.colorbar(phi_im, label='Phi')

    # q_im = axs[1].imshow(q[:,2,:], origin='lower',cmap='viridis',aspect='auto')
    # fig.colorbar(q_im, label='Q')

    # u_im = axs[0].imshow(u[:,2,:], origin='lower',cmap='viridis',aspect='auto')
    # fig.colorbar(u_im, label='u')

    # plt.show()

    # print("phi[0][0]:", phi[2][2])
    # print("bx[0][0]:", bx[2][2])
    # print("by[0][0]:", by[2][2])






# ********************************************************************************
# generate data
    # ********************************************************************************
# UQpol_array               = dg.uqpol_generator(U0=-0.1,Q0=0.1,box_length=8)
# d3_field                  = multigrid(UQpol_array)
# Ur, Qr, cos2gr, PAr, polr = UQ_reconstruct(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][2])

# barray      = dg.B_generator(bx0=1,by0=1,bz0=1,box_length=8) # lvls = bx,by,bz
# UQpol_array = dg.UQ_generator(60,boxsize=60,length=256,step=32) # lvls = U,Q,pol
# UQpol_array = dg.uqpol_generator(U0=0.2,Q0=0.1,box_length=64) # lvls = U,Q,pol
# UQpol_array = dg.wave_2d(amplitude=1, frequency=1, box_length=64)

# Plot resultant 3D field
# ********************************************************************************

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # First plot
# axs[0].plot(Ulist, PAi_list, marker='o', linestyle='-')
# axs[0].set_xlabel('U Value')
# axs[0].set_ylabel('Initial PA (degs)')
# axs[0].set_title('Box = 64, Q held constant at 0.1')
# axs[0].grid()
# # Second plot
# axs[1].imshow(residuals, extent=[Qlist.min(), Qlist.max(), Ulist.min(), Ulist.max()], origin='lower', aspect='auto', cmap='viridis')
# # axs[1].plot(Ulist, residual_list, marker='o', linestyle='-')
# axs[1].colorbar(label='Residual')
# axs[1].set_xlabel('U Value')
# axs[1].set_ylabel('Residual')
# axs[1].set_title('Residual')
# axs[1].grid()

# # third plot
# axs[2].plot(PAi_list, PAr_list, marker='o', linestyle='none')
# # axs[2].plot(PAi_list, PAi_list, linestyle='--', color='darkred')
# axs[2].set_xlabel('Initial PA (degs)')
# axs[2].set_ylabel('Reconstructed PA (degs)')
# axs[2].set_title('PAi vs PAr')
# axs[2].grid()
# # Adjust layout
# plt.tight_layout()
# plt.show()

# CLEAN UP PLOTTING FUNCTIONS BELOW:

# sk.visualize_3d_recon(field[-1][0], field[-1][1], field[-1][1], nskip=2, vlength=1, name='Reconstructed 3D Wavy Field')
# bx = field[-1][0]
# by = field[-1][1]
# bz = field[-1][2]
# nskip = 2
# vlength = 1
# name = 'Reconstructed 3D Wavy Field'


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Create a grid of points
# X, Y, Z = np.meshgrid(np.arange(bx.shape[0])[::nskip],
#                       np.arange(bx.shape[0])[::nskip],
#                       np.arange(bx.shape[0])[::nskip])
# # Plot every nth vector
# ax.quiver(X, Y, Z, bx[::nskip, ::nskip], by[::nskip, ::nskip], 
#             bz[::nskip, ::nskip], length=vlength, normalize=True, 
#             arrow_length_ratio=0)
# ax.set_xlabel('Bx')
# ax.set_ylabel('By')
# ax.set_zlabel('Bz')
# ax.set_title(name)
# plt.show()

# visualize_3d_vector(field[-1][0], field[-1][1], field[-1][2], n=16, leg=10)

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # # First plot
    # axs[0].plot(Ulist, PAi_list, marker='o', linestyle='-')
    # axs[0].set_xlabel('U Value')
    # axs[0].set_ylabel('Initial PA (degs)')
    # axs[0].set_title('Box = 64, Q held constant at 0.1')
    # axs[0].grid()
    # # Second plot
    # axs[1].imshow(residuals, extent=[Qlist.min(), Qlist.max(), Ulist.min(), Ulist.max()], origin='lower', aspect='auto', cmap='viridis')
    # # axs[1].plot(Ulist, residual_list, marker='o', linestyle='-')
    # axs[1].colorbar(label='Residual')
    # axs[1].set_xlabel('U Value')
    # axs[1].set_ylabel('Residual')
    # axs[1].set_title('Residual')
    # axs[1].grid()

    # # third plot
    # axs[2].plot(PAi_list, PAr_list, marker='o', linestyle='none')
    # # axs[2].plot(PAi_list, PAi_list, linestyle='--', color='darkred')
    # axs[2].set_xlabel('Initial PA (degs)')
    # axs[2].set_ylabel('Reconstructed PA (degs)')
    # axs[2].set_title('PAi vs PAr')
    # axs[2].grid()
    # # Adjust layout
    # plt.tight_layout()
    # plt.show()

# print(np.where(np.isnan(PAr_array))[0]) # PAr has some NaN



    # sk.visualize_PA_map(PAr)

    # Wavy Field; Loop thru Bz
    # ********************************************************************************
    # Bzlist    = np.linspace(-1,1,20)
    # residuals = np.zeros(len(Bzlist))
    # PAi_array = np.zeros(len(Bzlist))
    # PAr_array = np.zeros(len(Bzlist))

    # for i, iBz in enumerate(Bzlist):
    #     UQpol_array = dg.wave_3d(amplitude=1, frequency=1, box_length=16, z_contribution=iBz)
    #     d3_field    = multigrid(UQpol_array)
    #     Ur, Qr, cos2gr, PAr, polr = UQ_reconstruct(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][1])
    #     res         = residual(UQpol_array[0], UQpol_array[1], Ur, Qr)

    #     PAi     = np.mean(0.5*np.arctan2(UQpol_array[0],UQpol_array[1]))
    #     PAi_map = 0.5*np.arctan2(UQpol_array[0],UQpol_array[1])

    #     residuals[i] = res
    #     PAi_array[i] = PAi*(180/np.pi)
    #     PAr_array[i] = np.mean(PAr*(180/np.pi))


    # # plotting
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # ax[0].plot(Bzlist, residuals, marker='o', linestyle='none')
    # ax[0].set_xlabel('Bz value')
    # ax[0].set_ylabel('Residual')

    # im1 = ax[1].imshow(PAi_map, origin='lower', aspect='auto', cmap='viridis')
    # cbar1 = fig.colorbar(im1, ax=ax[1])
    # ax[1].set_title('Initial PA')
    # ax[1].set_xlabel('X')
    # ax[1].set_ylabel('Y')

    # im2 = ax[2].imshow(PAr, origin='lower', aspect='auto', cmap='viridis')
    # cbar2 = fig.colorbar(im2, ax=ax[2])
    # ax[2].set_title('Reconstructed PA')
    # ax[2].set_xlabel('X')
    # ax[2].set_ylabel('Y')

    # ax[1].set_xlabel('U Value')
    # ax[1].set_ylabel('Q Value', labelpad=0.01)
    # ax[1].grid()
    # cbar1.set_label(label='PAr', labelpad=0.01)

    # plt.show()


    # Uniform Field Loop:
    # ********************************************************************************
    # residual_list = []
    # PAi_list      = []
    # PAr_list      = []

    # Ulist = np.linspace(-1,1,20)
    # Qlist = np.linspace(-1,1,20)
    # residuals = np.zeros((len(Ulist), len(Qlist)))
    # PAi_array = np.zeros((len(Ulist), len(Qlist)))
    # PAr_array = np.zeros((len(Ulist), len(Qlist)))

    # for i, U in enumerate(Ulist):
    #     for j, Q in enumerate(Qlist):
    #         UQpol_array = dg.uqpol_generator(U0=U,Q0=Q,box_length=8)
    #         d3_field = multigrid(UQpol_array)
    #         Ur, Qr, cos2gr, PAr, polr = UQ_reconstruct(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][2])
    #         res = residual(UQpol_array[0], UQpol_array[1], Ur, Qr)
    #         PAi = np.mean(0.5*np.arctan2(UQpol_array[0],UQpol_array[1]))

    #         residuals[i][j] = res
    #         PAi_array[i][j] = PAi*(180/np.pi)
    #         PAr_array[i][j] = np.mean(PAr*(180/np.pi))


    # Plotting Uniform Field Loop:
    # ********************************************************************************
    # fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    # im0 = ax[0].imshow(PAi_array, extent=[Ulist.min(), Ulist.max(), Qlist.min(), Qlist.max()], origin='lower', aspect='auto', cmap='viridis')
    # ax[0].set_xlabel('U Value')
    # ax[0].set_ylabel('Q Value', labelpad=0.01)
    # ax[0].grid()
    # cbar0 = fig.colorbar(im0, ax=ax[0])
    # cbar0.set_label(label='PAi', labelpad=0.01)

    # im1 = ax[1].imshow(PAr_array, extent=[Ulist.min(), Ulist.max(), Qlist.min(), Qlist.max()], origin='lower', aspect='auto', cmap='viridis')
    # ax[1].set_xlabel('U Value')
    # ax[1].set_ylabel('Q Value', labelpad=0.01)
    # ax[1].grid()
    # cbar1 = fig.colorbar(im1, ax=ax[1])
    # cbar1.set_label(label='PAr', labelpad=0.01)

    # im2 = ax[2].imshow(residuals, extent=[Ulist.min(), Ulist.max(), Qlist.min(), Qlist.max()], origin='lower', aspect='auto', cmap='viridis')
    # ax[2].set_xlabel('U Value')
    # ax[2].set_ylabel('Q Value', labelpad=0.01)
    # ax[2].axhline(y=0, color='white', linestyle='-')
    # ax[2].axvline(x=0, color='white', linestyle='-')
    # ax[2].grid()
    # cbar2 = fig.colorbar(im2, ax=ax[2])
    # cbar2.set_label(label='Residual', labelpad=0.01)

    # sc = ax[3].scatter(PAi_array, PAr_array, c=residuals, cmap='viridis')
    # ax[3].set_xlabel('PAi Value')
    # ax[3].set_ylabel('PAr Value', labelpad=0.01)
    # ax[3].grid()
    # cbar3 = fig.colorbar(sc, ax=ax[3])
    # cbar3.set_label(label='Residual', labelpad=0.01)

    # plt.tight_layout()
    # plt.show()
    # ********************************************************************************
    # generate data
     # ********************************************************************************
    # UQpol_array               = dg.uqpol_generator(U0=-0.1,Q0=0.1,box_length=8)
    # d3_field                  = multigrid(UQpol_array)
    # Ur, Qr, cos2gr, PAr, polr = UQ_reconstruct(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][2])
    
    # barray      = dg.B_generator(bx0=1,by0=1,bz0=1,box_length=8) # lvls = bx,by,bz
    # UQpol_array = dg.UQ_generator(60,boxsize=60,length=256,step=32) # lvls = U,Q,pol
    # UQpol_array = dg.uqpol_generator(U0=0.2,Q0=0.1,box_length=64) # lvls = U,Q,pol
    # UQpol_array = dg.wave_2d(amplitude=1, frequency=1, box_length=64)

    # Plot resultant 3D field
    # ********************************************************************************

# def visual_UQ_xy(U,Q,label):
#     plt.figure(figsize=(8,8))
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Input PAs for a '+ label)
#     for i in range(len(U[0])):
#         for j in range(len(Q[0])):
#             iu = U[i][j]
#             iq = Q[i][j]
#             iphi = 0.5*np.arctan2(iu,iq)
#             x = np.sin(iphi)
#             y = np.cos(iphi)
#             plt.quiver(i, j, x, y, angles='xy', scale_units='xy', scale=1, headlength=0, headaxislength=0, headwidth=0)
#             # plt.quiver(i, j, x, y, headlength=0, headaxislength=0, headwidth=0)
#     plt.show()

# def visualize_slice_xy(x, y, label):
#     plt.figure(figsize=(8,8))
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Slice of Reconstructed '+ label)
#     for i in range(len(x[0])):
#         for j in range(len(y[0])):
#             ix = x[i][j]
#             iy = y[i][j]
#             # plt.quiver(i, j, ix, iy, angles='xy', scale_units='xy', scale=1, headlength=0, headaxislength=0, headwidth=0)
#             plt.quiver(i, j, ix, iy, headlength=0, headaxislength=0, headwidth=0)
#     plt.show()


# -----------------------------------------------------------
# plot reconstructed field in 2.5D
# -----------------------------------------------------------
def visualize_25d_old(bx, by, bz, name, plotdex):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.meshgrid(np.arange(bx.shape[0]), 
                          np.arange(by.shape[0]), 
                          np.arange(bz.shape[0]), indexing=plotdex)
    ax.quiver(X, Y, Z, bx, by, bz, length=0.5, normalize=True, 
            arrow_length_ratio=0, pivot='middle', color='dimgray')
    ax.set_xlabel('Bx')
    ax.set_ylabel('By')
    ax.set_zlabel('Bz')
    ax.set_title(name)
    plt.show()




# ================================================ # ================================================
# ================================================ # ================================================

# Plot Phi as colormap to make sure convert to stokes u/q good
# plt.figure(figsize=(8,8))
# plt.imshow(phi_array, interpolation='none')
# plt.colorbar(label='Phi')
# plt.title('2D Toroidal Phi Values')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.ylim(0, np.max(len(phi_array[0]))-1)
# plt.show()

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(X, Y, Z, bx, by, bz, length=0.25, normalize=True, arrow_length_ratio=0, pivot='middle', color='dodgerblue')
# ax.set_title("Input 3D Toroidal Vector Field")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()

# Plot slice of 3D field 
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('2D Slice of 3D Toroidal Vector Field from B Components')
# for i in range(len(X[0][0])):
#     for j in range(len(X[0][0])):
#         ax.quiver(X[i][j][0], Y[i][j][0], bx[i][j][0], by[i][j][0], headaxislength=0, headlength=0, headwidth=1, pivot='middle')
# plt.show()

# plot phi as vector field

# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('2D Pol Map of Toroidal Field')
# for i in range(len(X[0][0])):
#     for j in range(len(X[0][0])):
#         ax.quiver(X[i][j][0], Y[i][j][0], np.sin(phi_array[i][j]),np.cos(phi_array[i][j]), headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='dodgerblue')
# plt.show()

# PLOTTING HOURGLASS DEBUGGING:
        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.imshow(Bz_array, origin='lower', cmap='viridis',
        #         extent=[z.min(), z.max(), r.min(), r.max()], aspect='auto')
        # plt.colorbar(label='Bz')
        # plt.title('Bz Component')
        # plt.xlabel('z')
        # plt.ylabel('r')

        # plt.subplot(1, 2, 2)
        # plt.imshow(Br_array, origin='lower', cmap='plasma',
        #         extent=[z.min(), z.max(), r.min(), r.max()], aspect='auto')
        # plt.colorbar(label='Br')
        # plt.title('Br Component')
        # plt.xlabel('z')
        # plt.ylabel('r')

        # plt.tight_layout()
        # plt.show()

        # # === 2D Streamplot for Debugging ===
        # plt.figure(figsize=(6,6))
        # plt.streamplot(z, r, Bz_array.T, Br_array.T, color='k', linewidth=1)
        # plt.xlabel('z')
        # plt.ylabel('r')
        # plt.title('Field Lines in r-z Plane')
        # plt.gca().set_aspect('equal')
        # plt.tight_layout()
        # plt.show()
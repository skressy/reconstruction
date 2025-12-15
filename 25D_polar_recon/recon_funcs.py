##
# Reconstruction Functions and plotting for 2.5D Reconstruction Polarimetry Testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions Included:
#   init_recon_2D(u)
#   UQ_reconstruct_2D(bx,by,bz)
#   residual(U_i,Q_i,U_r,Q_r)
#   visualize_3d(bx, by, bz, name, plotdex)
#   visualize_gen_3D(X,Y,Z,bx,by,bz)
#   visualize_25d(bx, by, bz, name, plotdex)
#   visual_UQ(U,Q,label,plotdex)
#   visualize_slice(x, y, label, plotdex)
#   plot_slice_3views(X,Y,Z,bx,by,bz)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# =======================================================
import numpy as np
import matplotlib.pyplot as plt
# ===============================================
# Initial Reconstruction 2.5D from Stokes U/Q 
# ===============================================
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

# ===============================================
# Reconstruct 2.5D U,Q from Bx,By,Bz
# ===============================================
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

# ===============================================
# Calculate Residual
# ===============================================
def residual(U_i,Q_i,U_r,Q_r):
    resy = np.sqrt(np.mean((U_i - U_r)**2 + (Q_i - Q_r)**2))
    print(resy)
    return resy

# ===============================================
# plot reconstructed field in 3D
# ===============================================
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

# ===============================================
# plot generated field in 3D
# ===============================================
def visualize_gen_3D(X,Y,Z,bx,by,bz):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, bx, by, bz, length=0.05, normalize=True,linewidth=1.5, 
              arrow_length_ratio=0, color='dodgerblue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Magnetic Field: Hourglass Configuration')
    plt.tight_layout()
    plt.show()

# ===============================================
# Visualize 2.5D reconstructed field
# ===============================================
def visualize_25d(bx, by, bz, name, plotdex):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.meshgrid(np.arange(bx.shape[0]), 
                          np.arange(by.shape[0]), 
                          np.arange(bz.shape[0]), indexing=plotdex)
    
    ax.quiver(X,Y,Z,bx,by,bz,length=0.5, normalize=True, arrow_length_ratio=0, 
              pivot='middle', color='dimgray')
    ax.set_xlabel('Bx')
    ax.set_ylabel('By')
    ax.set_zlabel('Bz')
    ax.set_title(name)
    plt.show()

# ===============================================
# plot U/Q in 2D, functions for both mesh and xy coords
# ===============================================
def visual_UQ(U,Q,label,plotdex):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.set_title(label)
    X, Y = np.meshgrid(np.arange(U.shape[0]), np.arange(Q.shape[0]),indexing=plotdex)
    phi = 0.5*np.arctan2(U,Q)
    x = np.sin(phi)
    y = np.cos(phi)
    ax.quiver(X, Y, x, y, headaxislength=0, headlength=0, headwidth=1, 
              pivot='middle', color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

# ===============================================
# plot slice of 3D field ### DONT TOUCH!
# ===============================================
def visualize_slice(x, y, label, plotdex):
    X, Y = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[0]), 
                       indexing=plotdex)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.quiver(X, Y, x, y, headaxislength=0, headlength=0, headwidth=1, 
              pivot='middle', color='dimgray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Slice of Reconstructed '+ label)
    plt.axis('equal')
    plt.show()

# ===============================================
# plot Hourglass B field components for 3 axis
# ===============================================
def plot_slice_3views(X,Y,Z,bx,by,bz):
        fig, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].set_title('XY-Plane (Z ≈ 0)')
        axs[0].quiver(X[:, :, 0], Y[:, :, 0], bx[:, :, 0], by[:, :, 0],
                    headaxislength=0, headlength=0, headwidth=1, pivot='middle', 
                    color='dodgerblue')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].axis('equal')

        axs[1].set_title('ZX-Plane (Y ≈ 0)')
        axs[1].quiver(Z[0,:,:], X[0,:,:], bz[0,:,:], bx[0,:,:],
                headaxislength=0, headlength=0, headwidth=1, pivot='middle', 
                color='dodgerblue')
        axs[1].set_xlabel('Z')
        axs[1].set_ylabel('X')
        axs[1].axis('equal')

        axs[2].set_title('YZ-Plane (X ≈ 0)')
        axs[2].quiver(Y[:,0,:], Z[:,0,:], by[:,0,:], bz[:,0,:],
                headaxislength=0, headlength=0, headwidth=1, pivot='middle', 
                color='dodgerblue')
        axs[2].set_xlabel('Y')
        axs[2].set_ylabel('Z')
        axs[2].axis('equal')

        plt.tight_layout()
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
import recon_funcs as rf
from scipy.special import jn_zeros, j0, j1, erfc

# ================================================
# Generates 3D Uniform Field
# ================================================
def uniform_field(bx0,by0,bz0,box_length,visual):
    # for uniform field
    bx = np.full((box_length, box_length, box_length), bx0) 
    by = np.full((box_length, box_length, box_length), by0) 
    bz = np.full((box_length, box_length, box_length), bz0) 

    cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u             = 2*bx*by/(bx**2+by**2) * cos2g
    
    if visual == 1:
        rf.visualize_3d(bx, by, bz, name='Input 3D Uniform Field',plotdex='xy')
        rf.visual_UQ(u[0],q[0],label='Input Uniform Field',plotdex='xy')

    UQMAP = [u[0],q[0],cos2g[0]]
    return UQMAP

# uqs = uniform_field(bx0=0.5,by0=0.5,bz0=0,box_length=8,visual=1)

# ================================================
# Generates 3D Wavy Field
# ================================================
def wave_3d(amplitude, frequency, box_length, z_contribution, visual, plotdex):
    x = np.linspace(0, 2 * np.pi, box_length)
    y = np.linspace(0, 2 * np.pi, box_length)
    z = np.linspace(0, 2 * np.pi, box_length)
    X, Y, Z = np.meshgrid(x, y, z,indexing=plotdex) # stays xy

    # Generate single wave direction field
    angle = amplitude * np.cos(frequency * X)
    bx = np.cos(angle)
    by = np.sin(angle)
    bz = np.ones(Z.shape)*z_contribution

    cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u             = 2*bx*by/(bx**2+by**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    if visual == 1:
        rf.visualize_3d(bx, by, bz, name='Input Wavy Field',plotdex=plotdex)
        rf.visual_UQ(u[0],q[0],label='Wavy Field',plotdex='ij')
    
    UQMAP = [u[0], q[0], cos2g[0]]
    return  UQMAP 

# uqc = wave_3d(amplitude=0.5, frequency=1, box_length=8, z_contribution=0.5, visual=1)

# ================================================
# Generates 3D Wavy Field with inconsistent Zeeman Measurements. 
# Pol Map plotting also shows relative polarization %, or Z contribution.
# ================================================
def wave_3d_zeeman(amplitude, frequency, box_length, z_contribution, visual, plotdex):
    x = np.linspace(0, 2 * np.pi, box_length)
    y = np.linspace(0, 2 * np.pi, box_length)
    z = np.linspace(0, 2 * np.pi, box_length)
    X, Y, Z = np.meshgrid(x, y, z,indexing=plotdex) # stays xy

    # Generate single wave direction field
    angle = amplitude * np.cos(frequency * X)
    bx = np.cos(angle)
    by = np.sin(angle)
    # fill z space with zeros
    bz = 0*X

    # except make two points have zeeman measurements
    for i in range(Z.shape[2]):
        bz[box_length//2][box_length//4][i] = z_contribution
        bz[box_length//2][(box_length-1)-box_length//4][i] = z_contribution

    cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u             = 2*bx*by/(bx**2+by**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    if visual == 1:
        rf.visualize_3d(bx, by, bz, name='Input Wavy Field',plotdex=plotdex)
        rf.visual_UQ_z(u[:,:,0], q[:,:,0], cos2g[:,:,0],label='Input Wavy Field',plotdex=plotdex)
    
    UQMAP = [u[:,:,0], q[:,:,0], cos2g[:,:,0]]
    return  UQMAP 

# uqc = wave_3d_z(amplitude=0.5, frequency=1, box_length=8, z_contribution=2, visual=1, plotdex='xy')

# ================================================
# Generates 3D Toroidal field
# ================================================
def toroidal_field_3D(box_length, visual, alpha, plotdex):
    x = np.linspace(-1, 1, box_length)
    y = np.linspace(-1, 1, box_length)
    z = np.linspace(-1, 1, box_length)
    X, Y, Z = np.meshgrid(x, y, z, indexing=plotdex)

    r = np.sqrt(X**2 + Y**2) + 1e-5 

    bx = (-Y*np.cos(alpha))/r
    by = (X*np.cos(alpha))/r
    bz = np.sin(alpha) 

    cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u             = 2*bx*by/(bx**2+by**2) * cos2g

    phi_array = np.zeros((len(X[0][0]),len(X[0][0])))
    q_array   = np.zeros((len(X[0][0]),len(X[0][0])))
    u_array   = np.zeros((len(X[0][0]),len(X[0][0])))
    cos_array = np.zeros((len(X[0][0]),len(X[0][0])))
    
    for i in range(len(X[0][0])):
            for j in range(len(X[0][0])):
                q_array[i][j]   = q[i][j][0]
                u_array[i][j]   = u[i][j][0]
                cos_array[i][j] = cos2g[i][j][0]
                iphi = 0.5*np.arctan2(u[i][j][0],q[i][j][0])
                phi_array[i][j] = iphi

    if visual == 1:
        rf.visualize_3d(bx, by, bz, name='Input Toroidal Field',plotdex=plotdex)
        rf.visual_UQ(u_array,q_array,label='Toroidal Field',plotdex=plotdex)

    UQMAP = [u_array, q_array, cos_array]
    return UQMAP

# Example
# UQCOS = toroidal_field_3D(box_length=12, visual=1, alpha=0, plotdex='xy')

# ================================================
# Generates 3D Hourglass field
# ================================================
def hourglass_3d(visual, nskip, n, n_theta, axis):
    # n = 16          # resolution
    # n_theta = 16    # angular resolution
    R_outer = 5.0
    h = 1.0
    B0 = 1.0
    Bm = 1.0
    betam = Bm / B0
    N_terms = 3    # reduced to avoid overflow

    # Grids (dimensionless: r, z ∈ [-2, 2])
    r = np.linspace(-0.25, 0.25, n)
    z = np.linspace(-0.25, 0.25, n)
    R_grid, Z_grid = np.meshgrid(r, z, indexing='xy')

    # Precompute Bessel roots and weights
    roots = jn_zeros(1, N_terms)          # λ_n (zeros of J1)
    weights = 2 / (roots * j0(roots)**2)  # weighting for summation
    eta = h / R_outer

    # Initialize field arrays
    Br_array = np.zeros_like(R_grid)
    Bz_array = np.zeros_like(R_grid)

    # === Compute Br and Bz ===
    for i in range(n):
        for j in range(n):
            ir = r[i]
            iz = z[j]

            J0_eval = j0(roots * ir)
            J1_eval = j1(roots * ir)

            arg_erfc_neg = roots * eta / 2 - iz / eta
            arg_erfc_pos = roots * eta / 2 + iz / eta

            ernie_neg = erfc(arg_erfc_neg)
            ernie_pos = erfc(arg_erfc_pos)

            exp_neg = np.exp(-roots * iz)
            exp_pos = np.exp(roots * iz)

            brackets0 = ernie_pos * exp_pos + ernie_neg * exp_neg
            brackets1 = ernie_neg * exp_neg - ernie_pos * exp_pos

            Br = np.sum(weights * J1_eval * betam * brackets1)
            Bz = np.sum(weights * J0_eval * betam * brackets0) + 1

            Br_array[i, j] = Br / B0
            Bz_array[i, j] = Bz / B0

    # conversion to cartesian coords!!
    theta = np.linspace(0, 2 * np.pi, n_theta)
    R2D, T2D, Z3D = np.meshgrid(r, theta, z, indexing='xy')

    X = R2D * np.cos(T2D)
    Y = R2D * np.sin(T2D)

    # create magnetic field components
    Br3D = np.tile(Br_array[np.newaxis, :, :], (n_theta, 1, 1))
    Bz3D = np.tile(Bz_array[np.newaxis, :, :], (n_theta, 1, 1))

    # convert to cartesian
    bx_cart = Br3D * np.cos(T2D)
    by_cart = Br3D * np.sin(T2D)
    bz_cart = Bz3D

    pos_array = np.array([X,Y,Z3D])
    b_array    = np.array([bx_cart,by_cart,bz_cart])

    i_inds = [0, 2, 1] 
    j_inds = [1, 0, 2]
    k_inds = [2, 1, 0]

    i = i_inds[axis]
    j = j_inds[axis]
    k = k_inds[axis]

    bx = b_array[i] #z #y
    by = b_array[j] #x #z
    bz = b_array[k] #y #x

    i_pos = pos_array[i]
    j_pos = pos_array[j]
    k_pos = pos_array[k]

    cos2g = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q     = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u     = 2*bx*by/(bx**2+by**2) * cos2g
    phi   = 0.5*np.arctan2(u,q)

    bi_stokes = np.sin(phi)
    bj_stokes = np.cos(phi)

    # slice correctly depending on axis chosen.
    if axis == 0:
        UQMAP = [u[:,:,0], q[:,:,0], cos2g[:,:,0]]
    elif axis == 1:
        UQMAP = [u[0,:,:], q[0,:,:], cos2g[0,:,:]]
    elif axis == 2:
        UQMAP = [u[:,0,:], q[:,0,:], cos2g[:,0,:]]
    
    
    if visual == 1:
        # plot 3D structure
        rf.visualize_gen_3D(X,Y,Z3D,bx_cart,by_cart,bz_cart)
        # plot slice of 3D structure
        rf.plot_slice_3views(X,Y,Z3D,bx_cart,by_cart,bz_cart)
        # plot stokes map
        plt.figure(figsize=(5,5))
        if axis == 0:
            plt.title('XY-Plane (Z ≈ 0)')
            plt.quiver(i_pos[:,:,0], j_pos[:,:,0], bi_stokes[:,:,0], bj_stokes[:,:,0],
                        headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
            plt.xlabel('X')
            plt.ylabel('Y')
        elif axis == 1:
            plt.title('ZX-Plane (Y ≈ 0)')
            plt.quiver(i_pos[0,:,:], j_pos[0,:,:], bi_stokes[0,:,:], bj_stokes[0,:,:],
                        headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
            plt.xlabel('Z')
            plt.ylabel('X') 
        elif axis == 2:
            plt.title('YZ-Plane (X ≈ 0)')
            plt.quiver(i_pos[:,0,:], j_pos[:,0,:], bi_stokes[:,0,:], bj_stokes[:,0,:],
                        headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
            plt.xlabel('Y')
            plt.ylabel('Z') 
        plt.tight_layout()
        plt.axis('equal')
        plt.show()

    return UQMAP

# uqmap = hourglass_3d(visual=1,nskip=1,n=8,n_theta=8,axis=2)

# ================================================
# Generates 3D LOS wavy field
# ================================================
def bz_wavy_field(amplitude, frequency, box_length, y_const, visual, plotdex):
    x = np.linspace(0, 2 * np.pi, box_length)
    y = np.linspace(0, 2 * np.pi, box_length)
    z = np.linspace(0, 2 * np.pi, box_length)
    X, Y, Z = np.meshgrid(x, y, z,indexing=plotdex) # stays xy

    # Wavy along LOS
    # angle = amplitude * np.cos(frequency * Z)
    # bx = np.sin(angle)
    # by = y_const
    # bz = np.cos(angle)

    # Wave perpendicular to LOS
    angle = amplitude * np.cos(frequency * Y)
    bx = y_const
    bz = np.sin(angle)
    by = np.cos(angle)
    print(bz)

    cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u             = 2*bx*by/(bx**2+by**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    if visual == 1:
        rf.visualize_3d(bx, by, bz, name='Input Wavy Field',plotdex=plotdex)
        rf.visual_UQ_depol(u[:,:,0], q[:,:,0], cos2g[:,:,0],label='Input Wavy Field',plotdex=plotdex)
    
    UQMAP = [u[:,:,0], q[:,:,0], cos2g[:,:,0]]
    return  UQMAP 

bz_wavy_field(amplitude=1, frequency=1, box_length=8, y_const=0, visual=1, plotdex='xy')
# ================================================ # ================================================
# ================================================ # ================================================
                                    # FUNCTIONS NOT USED BELOW #
# ================================================ # ================================================
# ================================================ # ================================================


# ================================================
# Generates U, Q, cos2g components, all uniform/constant
# ================================================
def uq_generator(U0,Q0,box_length,visual):
    # for uniform field
    U_map = np.full((box_length, box_length), U0) 
    Q_map = np.full((box_length, box_length), Q0) 
    pol = np.sqrt(U0**2 + Q0**2)
    pol_map = np.full((box_length, box_length), pol) 
    if visual == 1:
        rf.visual_UQ(U_map,Q_map,label='Uniform Field',plotdex='xy')
    UQMAP = [U_map,Q_map,pol_map]
    return UQMAP

# uqpol_generator(0.1,0.3,16,visual=1)

# ================================================
# generates 2D sinusoidal field
# ================================================
def sinusoidal_2d(amplitude, frequency, box_length):
    x = np.linspace(0, 2 * np.pi, box_length)
    y = np.linspace(0, 2 * np.pi, box_length)
    X, Y = np.meshgrid(x, y,indexing='ij')

    # Generate sinusoidal vector field
    bx = amplitude * np.sin(frequency * X)
    by = amplitude * np.cos(frequency * Y)

    return bx, by

# ================================================
# generates 3D sinusoidal field
# ================================================
def sinusoidal_3d(amplitude, frequency, box_length, z_value):
    x = np.linspace(0, 2 * np.pi, box_length)
    y = np.linspace(0, 2 * np.pi, box_length)
    z = np.linspace(0, 2 * np.pi, box_length)
    X, Y, Z = np.meshgrid(x, y, z,indexing='ij')

    # Generate sinusoidal vector field
    bx = amplitude * np.sin(frequency * X)
    by = amplitude * np.sin(frequency * Y)
    bz = np.full_like(X, z_value)

    return bx, by, bz
# ================================================
# generates 2D Toroidal field
# ================================================
def toroidal_field_2D(box_length, visual):
    x = np.linspace(-1, 1, box_length)
    y = np.linspace(-1, 1, box_length)
    X, Y = np.meshgrid(x, y, indexing='xy')

    r = np.sqrt(X**2 + Y**2) + 1e-5  # Add a small value to avoid division by zero

    bx = -Y / r
    by = X / r   
    bz = 0

    cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u             = 2*bx*by/(bx**2+by**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)

    if visual == 1:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.quiver(X, Y, bx, by, headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Toroidal Field')
        plt.axis('equal')
        plt.show()

        rf.visual_UQ(u,q,label='Toroidal Field',plotdex='xy') # this works

        plt.imshow(phi, interpolation='none')
        plt.colorbar(label='phi')
        plt.title('2D Circular (Toroidal) Phi Values')
        plt.ylim(0, np.max(len(phi[0]))-1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return [u,q,cos2g]

# Example usage
# uqcarray = toroidal_field_2D(box_length=16, visual=1)

# ================================================
# Generates a single wave field in 2D
# assumes Bz component is 0, finds Bx, By
# returns Stokes U, Q, cos2g
# ================================================
def wave_2d(amplitude, frequency, box_length):
    x = np.linspace(0, 2 * np.pi, box_length)
    y = np.linspace(0, 2 * np.pi, box_length)
    X, Y = np.meshgrid(x, y, indexing='xy') # stays xy

    # Generate single wave direction field
    angle = amplitude * np.sin(frequency * X)
    bx = np.cos(angle)
    by = np.sin(angle)
    bz = 0 # no Z component in 2D

    cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
    q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
    u             = 2*bx*by/(bx**2+by**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    fig, ax = plt.subplots()
    nskip   = 1
    # Create a grid of points
    X, Y = np.meshgrid(np.arange(bx.shape[0])[::nskip],
                       np.arange(bx.shape[0])[::nskip],indexing='xy')
    # Plot every nth vector
    ax.quiver(X, Y, bx[::nskip, ::nskip], by[::nskip, ::nskip],headaxislength=0,headlength=0,headwidth=1)
    ax.set_xlabel('Bx')
    ax.set_ylabel('By')
    ax.set_title('Input 3D Wavy Field')
    plt.show()

    UQMAP = [u,q,cos2g]
    return UQMAP

# wave_2d(1,1,10)

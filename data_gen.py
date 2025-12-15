import numpy as np
import matplotlib.pyplot as plt
import math
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
import recon_funcs as rf
from scipy.special import jn_zeros, j0, j1, erfc
import random

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
        rf.visual_UQ(u[:,:,0],q[:,:,0],label='Input Wavy Field',plotdex='ij')
    
    UQMAP = [u[:,:,0], q[:,:,0], cos2g[:,:,0]]
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
        rf.visual_UQ_depol(u[:,:,0], q[:,:,0], cos2g[:,:,0],label='Input Wavy Field',plotdex=plotdex)
    
    UQMAP = [u[:,:,0], q[:,:,0], cos2g[:,:,0]]
    return  UQMAP 

# uqc = wave_3d_zeeman(amplitude=0.5, frequency=1, box_length=8, z_contribution=2, visual=1, plotdex='xy')

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

# bz_wavy_field(amplitude=1, frequency=1, box_length=8, y_const=0, visual=1, plotdex='xy')

# ================================================
# Generates 3D LOS Torodial field
# ================================================
def torodial_zeeman_pol(box_length,alpha,plotting):

    x = np.linspace(-1, 1, box_length)
    y = np.linspace(-1, 1, box_length)
    z = np.linspace(-1, 1, box_length)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

    # field geometry - Torodial
    r = np.sqrt(Y**2 + Z**2) + 1e-5 
    Bx = np.sin(alpha) 
    By = (-Z*np.cos(alpha))/r 
    Bz = (Y*np.cos(alpha))/r  

    # calculate Stokes U/Q/Cos2g
    cos2g         = (Bx**2+By**2)/(Bx**2+By**2+Bz**2)
    q             = (By**2-Bx**2)/(Bx**2+By**2) * cos2g
    u             = 2*Bx*By/(Bx**2+By**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    los = np.array([0., 0., 1.])   # unit vector; change as desired
    los = los / np.linalg.norm(los)

    # compute B_parallel field: pointwise dot product
    B_parallel_3d = los[0]*Bx + los[1]*By + los[2]*Bz

    # integrate or average along LOS axis, along z, the axis to collapse is index 2 (the third axis).
    B_los_map = np.mean(B_parallel_3d, axis=2)

    # grab first slice of Stokes U/Q/COS2G
    U = u[:,:,0]
    Q = q[:,:,0]
    COS = cos2g[:,:,0]

    if plotting == 1:
        # Plot B_los_map
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        im0 = ax[0].imshow(B_los_map, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
        ax[0].set_title(r'B$_{LOS}$ (z)')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

        # plot stokes U and Q Map
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]), indexing='ij')
        factor = 1 / np.max(COS)

        phi = 0.5 * np.arctan2(U, Q)
        x_scaled = np.sin(phi) * COS * factor
        y_scaled = np.cos(phi) * COS * factor

        ax[1].set_title('Stokes U and Q Map')
        ax[1].quiver(Y,X,y_scaled,x_scaled,scale=2, scale_units='xy', headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')

        plt.tight_layout()
        plt.show()

        step = 1
        rf.visualize_3d(Bx, By, Bz, name='Input Wavy Field',plotdex='xy')

    return U, Q, COS, B_los_map, Bx, By, Bz

# u, q, cos, blos, bx, by, bz = torodial_zeeman_pol(box_length=16,alpha=0,plotting=1)
# print(u, q, cos, blos)

# ZEEMAN ONLY BELOW

# ================================================
# Generates polarization AND Zeeman maps
# ================================================
def wavy_zeeman_pol(box_length,amplitude,frequency,x_const,plotting):

    x = np.linspace(0, 2*np.pi, box_length)
    y = np.linspace(0, 2*np.pi, box_length)
    z = np.linspace(0, 2*np.pi, box_length)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

    # field geoemtry - wavy
    angle = amplitude * np.cos(frequency * Y)
    Bx = np.full_like(X, x_const)      
    Bz = np.sin(angle)                 
    By = np.cos(angle)

    # calculate Stokes U/Q/Cos2g
    cos2g         = (Bx**2+By**2)/(Bx**2+By**2+Bz**2)
    q             = (By**2-Bx**2)/(Bx**2+By**2) * cos2g
    u             = 2*Bx*By/(Bx**2+By**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    los = np.array([0., 0., 1.])   # unit vector; change as desired
    los = los / np.linalg.norm(los)

    # compute B_parallel field: pointwise dot product
    B_parallel_3d = los[0]*Bx + los[1]*By + los[2]*Bz

    # integrate or average along LOS axis, along z, the axis to collapse is index 2 (the third axis).
    B_los_map = np.mean(B_parallel_3d, axis=2)

    # grab first slice of Stokes U/Q/COS2G
    U = u[:,:,0]
    Q = q[:,:,0]
    COS = cos2g[:,:,0]

    if plotting == 1:
        # Plot B_los_map
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        im0 = ax[0].imshow(B_los_map, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
        ax[0].set_title(r'B$_{LOS}$ (z)')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

        # plot stokes U and Q Map
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]), indexing='ij')
        factor = 1 / np.max(COS)

        phi = 0.5 * np.arctan2(U, Q)
        x_scaled = np.sin(phi) * COS * factor
        y_scaled = np.cos(phi) * COS * factor

        ax[1].set_title('Stokes U and Q Map')
        ax[1].quiver(Y,X,y_scaled,x_scaled,scale=2, scale_units='xy', headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')

        plt.tight_layout()
        plt.show()

        # step = 3
        # rf.visualize_3d(Bx[::step, ::step, ::step], By[::step, ::step, ::step], Bz[::step, ::step, ::step], name='Input Wavy Field',plotdex='xy')

    return U, Q, COS, B_los_map, Bx, By, Bz

# ================================================
# Generates polarization AND Zeeman maps with FEW constant Zeeman measurements (to create realisitic observations)
# ================================================
def wavy_zeeman_pol_min(box_length,amplitude,frequency,x_const,num_measurements,plotting):
    
    nz = num_measurements

    x = np.linspace(0, 2*np.pi, box_length)
    y = np.linspace(0, 2*np.pi, box_length)
    z = np.linspace(0, 2*np.pi, box_length)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

    # field geoemtry - wavy
    # angle = amplitude * np.cos(frequency * Y)
    # Bx = np.full_like(X, x_const)      
    # Bz = np.sin(angle)                 
    # By = np.cos(angle)

    angle = amplitude * np.cos(frequency * X)
    Bx = np.sin(angle)
    By = np.cos(angle)
    Bz = np.full_like(X, x_const)  # fill z space with zeros

    # calculate Stokes U/Q/Cos2g
    cos2g         = (Bx**2+By**2)/(Bx**2+By**2+Bz**2)
    q             = (By**2-Bx**2)/(Bx**2+By**2) * cos2g
    u             = 2*Bx*By/(Bx**2+By**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    los = np.array([0., 0., 1.])   # unit vector; change as desired
    los = los / np.linalg.norm(los)

    # compute B_parallel field: pointwise dot product
    B_parallel_3d = los[0]*Bx + los[1]*By + los[2]*Bz

    # integrate or average along LOS axis, along z, the axis to collapse is index 2 (the third axis).
    B_los_map = np.mean(B_parallel_3d, axis=2)

    # grab first slice of Stokes U/Q/COS2G
    U = u[:,:,0]
    Q = q[:,:,0]
    COS = cos2g[:,:,0]

    if plotting == 1:
        # Plot B_los_map
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        im0 = ax[0].imshow(B_los_map, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
        ax[0].set_title(r'B$_{LOS}$ (z)')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

        # plot stokes U and Q Map
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]), indexing='ij')
        factor = 1 / np.max(COS)

        phi = 0.5 * np.arctan2(U, Q)
        x_scaled = np.sin(phi) * COS * factor
        y_scaled = np.cos(phi) * COS * factor

        ax[1].set_title('Stokes U and Q Map')
        ax[1].quiver(Y,X,y_scaled,x_scaled,scale=2, scale_units='xy', headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')

        plt.tight_layout()
        plt.show()

        step = 2
        rf.visualize_3d(Bx[::step, ::step, ::step], By[::step, ::step, ::step], Bz[::step, ::step, ::step], name='Input Wavy Field',plotdex='ij')
    return U, Q, COS, B_los_map[:nz,:nz], Bx, By, Bz

# U, Q, COS, B_los_map, Bx, By, Bz = wavy_zeeman_pol_min(box_length=16,amplitude=1.2,frequency=1,x_const=0.2,num_measurements=4,plotting=1)


# # ================================================================
# # Generates polarization AND Zeeman maps with FEW 
# # VARIABLE Zeeman measurements (to create realisitic observations)
# # ================================================================
# def wavy_zeeman_pol_min_vary(box_length,amplitude,frequency,z_vals, num_measurements,plotting):
    
#     nz = num_measurements

#     x = np.linspace(0, 2*np.pi, box_length)
#     y = np.linspace(0, 2*np.pi, box_length)
#     z = np.linspace(0, 2*np.pi, box_length)
#     X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

#     angle = amplitude * np.cos(frequency * X)
#     Bx = np.sin(angle)
#     By = np.cos(angle)
#     Bz = np.full_like(X, fill_value=0)  # fill z space with zeros

#     Bz[0:int(0.5*box_length),0:int(0.5*box_length),:] = z_vals[0]
#     Bz[int(0.5*box_length):,0:int(0.5*box_length),:] = z_vals[1]
#     Bz[int(0.5*box_length):,int(0.5*box_length):,:] = z_vals[2]
#     Bz[0:int(0.5*box_length),int(0.5*box_length):,:] = z_vals[3]

#     print(np.shape(Bz))

#     if plotting == 1:
#         fig, ax = plt.subplots(1,1, figsize=(4,4))
#         im = ax.imshow(Bz[:,:,0], origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
#         ax.set_title(r'B$_z$')
#         plt.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
#         plt.show()

#     # calculate Stokes U/Q/Cos2g
#     cos2g         = (Bx**2+By**2)/(Bx**2+By**2+Bz**2)
#     q             = (By**2-Bx**2)/(Bx**2+By**2) * cos2g
#     u             = 2*Bx*By/(Bx**2+By**2) * cos2g
#     phi           = 0.5*np.arctan2(u,q)
#     pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

#     los = np.array([0., 0., 1.])   # unit vector; change as desired
#     los = los / np.linalg.norm(los)

#     # compute B_parallel field: pointwise dot product
#     B_parallel_3d = los[0]*Bx + los[1]*By + los[2]*Bz

#     # integrate or average along LOS axis, along z, the axis to collapse is index 2 (the third axis).
#     B_los_map = np.mean(B_parallel_3d, axis=2)

#     # grab first slice of Stokes U/Q/COS2G
#     U = u[:,:,0]
#     Q = q[:,:,0]
#     COS = cos2g[:,:,0]


#     # # create array full of markers (-1) to say no Zeeman measurement here!
#     # minBLOS = np.full_like(BLOS, fill_value=-1)

#     # # For some user-specified number of zeeman measurements, fill in minBLOS based on BLOS
#     # for it in range(n_zs):
#     #     rand_i = np.random(nx)
#     #     rand_j = np.random(ny)
#     #     minBLOS[rand_i][rand_j] = BLOS[rand_i][rand_j]

#     if plotting == 1:
#         # Plot B_los_map
#         fig, ax = plt.subplots(1,2, figsize=(8,4))
#         im0 = ax[0].imshow(B_los_map, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
#         ax[0].set_title(r'B$_{LOS}$ (z)')
#         ax[0].set_xlabel('x')
#         ax[0].set_ylabel('y')
#         plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

#         # plot stokes U and Q Map
#         X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]), indexing='ij')
#         factor = 1 / np.max(COS)

#         phi = 0.5 * np.arctan2(U, Q)
#         x_scaled = np.sin(phi) * COS * factor
#         y_scaled = np.cos(phi) * COS * factor

#         ax[1].set_title('Stokes U and Q Map')
#         ax[1].quiver(Y,X,y_scaled,x_scaled,scale=2, scale_units='xy', headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
#         ax[1].set_xlabel('X')
#         ax[1].set_ylabel('Y')

#         plt.tight_layout()
#         plt.show()

#         step = 2
#         rf.visualize_3d(Bx[::step, ::step, ::step], By[::step, ::step, ::step], Bz[::step, ::step, ::step], name='Input Wavy Field',plotdex='ij')
#     return U, Q, COS, B_los_map[:nz,:nz], Bx, By, Bz

# U, Q, COS, B_los_map, Bx, By, Bz = wavy_zeeman_pol_min_vary(box_length=16,amplitude=1.2,frequency=1,z_vals=[1,2,3,4],num_measurements=4,plotting=1)

# ================================================================
# Generates polarization AND Zeeman maps with FEW 
# VARIABLE Zeeman measurements (to create realisitic observations)
# ================================================================
def wavy_zeeman_pol_min_vary(box_length,amplitude,frequency,z_vals, nz, plotting):
    
    x = np.linspace(0, 2*np.pi, box_length)
    y = np.linspace(0, 2*np.pi, box_length)
    z = np.linspace(0, 2*np.pi, box_length)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

    # Bx, By Geometry
    angle = amplitude * np.cos(frequency * X)
    Bx = np.sin(angle)
    By = np.cos(angle)

    # Bz Geometry
    Bz = np.full_like(X, fill_value=0)  # fill z space with zeros
    Bz[0:int(0.5*box_length),0:int(0.5*box_length),:] = z_vals[0]
    Bz[int(0.5*box_length):,0:int(0.5*box_length),:] = z_vals[1]
    Bz[int(0.5*box_length):,int(0.5*box_length):,:] = z_vals[2]
    Bz[0:int(0.5*box_length),int(0.5*box_length):,:] = z_vals[3]

    # plot a slice of Bz to verify
    if plotting == 1:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
        im = ax.imshow(Bz[:,:,0], origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
        ax.set_title(r'B$_z$')
        plt.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
        plt.show()

    # calculate Stokes U/Q/Cos2g
    cos2g         = (Bx**2+By**2)/(Bx**2+By**2+Bz**2)
    q             = (By**2-Bx**2)/(Bx**2+By**2) * cos2g
    u             = 2*Bx*By/(Bx**2+By**2) * cos2g
    phi           = 0.5*np.arctan2(u,q)
    pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

    los = np.array([0., 0., 1.])   # unit vector; change as desired
    los = los / np.linalg.norm(los)

    # compute B_parallel field: pointwise dot product
    B_parallel_3d = los[0]*Bx + los[1]*By + los[2]*Bz

    # integrate or average along LOS axis, along z, the axis to collapse is index 2 (the third axis).
    B_los_map = np.mean(B_parallel_3d, axis=2)

    # grab first slice of Stokes U/Q/COS2G
    U = u[:,:,0]
    Q = q[:,:,0]
    COS = cos2g[:,:,0]

    # Reduce B_los_map to only have nz measurements
    # create array full of markers (-1) to say no Zeeman measurement here!
    minBLOS = np.full_like(B_los_map, fill_value=-1)

    nx, ny = B_los_map.shape[0], B_los_map.shape[1]
    # For some user-specified number of zeeman measurements, fill in minBLOS based on BLOS
    if nz <= 0:
        'No Random Zeeman Measurements. Returning empty minBLOS map.'
    else:
        for it in range(nz):
            rand_i = random.randint(0,nx-1)
            rand_j = random.randint(0,ny-1)
            minBLOS[rand_i][rand_j] = B_los_map[rand_i][rand_j]

    # PLot BLOS, stokes U/Q, vectors
    if plotting == 1:
        # Plot B_los_map
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        im0 = ax[0].imshow(B_los_map, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
        ax[0].set_title(r'B$_{LOS}$ (z)')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(minBLOS, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
        ax[1].set_title(r'Observed B$_{LOS}$ (z)')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        plt.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)

        # plot stokes U and Q Map
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]), indexing='ij')
        factor = 1 / np.max(COS)
        # plot vectors
        phi = 0.5 * np.arctan2(U, Q)
        x_scaled = np.sin(phi) * COS * factor
        y_scaled = np.cos(phi) * COS * factor

        ax[2].set_title('Stokes U and Q Map')
        ax[2].quiver(Y,X,y_scaled,x_scaled,scale=2, scale_units='xy', headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue')
        ax[2].set_xlabel('X')
        ax[2].set_ylabel('Y')

        plt.tight_layout()
        plt.show()

        step = 2
        rf.visualize_3d(Bx[::step, ::step, ::step], By[::step, ::step, ::step], Bz[::step, ::step, ::step], name='Input Wavy Field',plotdex='ij')
    
    return U, Q, COS, minBLOS, Bx, By, Bz

# U, Q, COS, B_los_map, Bx, By, Bz = wavy_zeeman_pol_min_vary(box_length=16,amplitude=1.2,frequency=1,z_vals=[1,2,3,4],nz=4,plotting=1)
# U, Q, COS, B_los_map, Bx, By, Bz = wavy_zeeman_pol_min_vary(box_length=16,amplitude=1.2,frequency=1,z_vals=[1,2,3,4],nz=0,plotting=1)


# ==================================================================# ==================================================================
# ==================================================================# ==================================================================
# ==================================================================# ==================================================================
# ========================================================== Work In-Progress ==========================================================
# ==================================================================# ==================================================================
# ==================================================================# ==================================================================
# ==================================================================# ==================================================================

# def check_chatgpt_recon():
#     # -------------------------------
#     # Example input arrays
#     # (replace these with your own Q, U, cos2gamma, Blos maps)
#     # -------------------------------
#     nx, ny = 36, 36
#     x = np.linspace(0, 2*np.pi, nx)
#     y = np.linspace(0, 2*np.pi, ny)
#     X, Y = np.meshgrid(x, y, indexing="xy")

#     # Toy model inputs
#     Q = np.ones((nx, ny))       # constant Q
#     U = np.zeros((nx, ny))      # constant U → polarization angle = 0
#     Cos2gamma = np.full((nx, ny), 0.5)   # constant inclination
#     Blos =  10* np.cos(Y)   # LOS variation only

#     # -------------------------------
#     # Step 1. Plane-of-sky orientation from Q,U
#     # -------------------------------
#     phi = 0.5 * np.arctan2(U, Q)   # polarization angle

#     # -------------------------------
#     # Step 2. Field magnitude from cos²γ and Zeeman
#     # cos²γ = (bx^2+by^2)/(bx^2+by^2+bz^2) = (B_pos^2) / (B^2)
#     # sin²γ = 1 - cos²γ
#     # cosγ = B_pos/B
#     # sinγ = B_los/B
#     # sin²γ = 1 - cos²γ = B_los^2/B^2
#     # B^2 = B_los^2/1 - cos²γ
#     # B = B_los/sqrt(1 - cos²γ)
#     # -------------------------------
#     Bmag = Blos / np.sqrt(1 - np.clip(Cos2gamma, 1e-6, 1.0))

#     # -------------------------------
#     # Step 3. POS amplitude
#     # -------------------------------
#     Bperp = np.sqrt(np.maximum(Bmag**2 - Blos**2, 0))

#     # -------------------------------
#     # Step 4. LOS component from Zeeman directly
#     # -------------------------------
#     bz = Blos.copy()
    
#     # -------------------------------
#     # Step 5. Scale bx, by
#     # -------------------------------
#     bx = np.cos(phi) * Bperp
#     by = np.sin(phi) * Bperp

#     # -------------------------------
#     # Diagnostics / plotting
#     # -------------------------------
#     fig, ax = plt.subplots(1, 5, figsize=(22,4))

#     im0 = ax[0].imshow(bx, origin="lower", cmap="RdBu")
#     ax[0].set_title("Bx map")
#     plt.colorbar(im0, ax=ax[0],fraction=0.046, pad=0.04)

#     im1 = ax[1].imshow(by, origin="lower", cmap="RdBu")
#     ax[1].set_title("By map")
#     plt.colorbar(im1, ax=ax[1],fraction=0.046, pad=0.04)

#     im2 = ax[2].imshow(bz, origin="lower", cmap="RdBu")
#     ax[2].set_title("Bz map (Zeeman)")
#     plt.colorbar(im2, ax=ax[2],fraction=0.046, pad=0.04)

#     Btot = np.sqrt(bx**2 + by**2 + bz**2)
#     im3 = ax[3].imshow(Btot, origin="lower", cmap="viridis")
#     ax[3].set_title("|B| total magnitude")
#     plt.colorbar(im3, ax=ax[3],fraction=0.046, pad=0.04)

#     # Quiver plot: POS field vectors
#     step = 1   # downsample arrows
#     ax[4].imshow(Btot, origin="lower", cmap="gray")
#     i = np.arange(0, nx, step)
#     j = np.arange(0, ny, step)
#     ax[4].quiver(i, j, bx[::step, ::step], by[::step, ::step],
#                 color="red", headaxislength=0, headlength=0, headwidth=1, pivot='middle')
#     ax[4].set_title("POS field (bx, by)")

#     plt.tight_layout()
#     plt.show()

#     step = 3
#     rf.visualize_3d(bx[::step, ::step], by[::step, ::step], bz[::step, ::step], name='Input Wavy Field',plotdex='xy')

# check_chatgpt_recon()



# # ==================================================================
# # Generates Synthetic Polarization and Zeeman data for a Wavy Field
# # ==================================================================
# def zeeman_wvy(box_length,amplitude,frequency,y_const):
#     # --------------------------
#     # parameters / geometry
#     # --------------------------
#     x = np.linspace(0, 2*np.pi, box_length)
#     y = np.linspace(0, 2*np.pi, box_length)
#     z = np.linspace(0, 2*np.pi, box_length)
#     X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

#     # field geoemtry - wavy
#     angle = amplitude * np.cos(frequency * Y)
#     Bx = np.full_like(X, y_const)      # constant in x
#     Bz = np.sin(angle)                 # varies with Y
#     By = np.cos(angle)

#     # calculate Stokes U/Q/Cos2g
#     cos2g         = (Bx**2+By**2)/(Bx**2+By**2+Bz**2)
#     q             = (By**2-Bx**2)/(Bx**2+By**2) * cos2g
#     u             = 2*Bx*By/(Bx**2+By**2) * cos2g
#     phi           = 0.5*np.arctan2(u,q)
#     pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

#     # --------------------------
#     # choose observer direction (LOS)
#     # example: observer along +z (i.e. look down z-axis)
#     # --------------------------
#     los = np.array([0., 0., 1.])   # unit vector; change as desired
#     los = los / np.linalg.norm(los)

#     # compute B_parallel field: pointwise dot product
#     B_parallel_3d = los[0]*Bx + los[1]*By + los[2]*Bz   # shape (nx,ny,nz)

#     # integrate or average along LOS axis.
#     # If LOS is along z, the axis to collapse is index 2 (the third axis).
#     # We detect which axis corresponds to LOS for simplicity: here we assume grid axes are (x,y,z)
#     # So if los roughly aligns with z, collapse axis=2. For arbitrary LOS you'd sample ray lines.
#     # For this toy case we'll assume LOS==z and average along axis=2:
#     B_los_map = np.mean(B_parallel_3d, axis=2)   # shape (nx, ny)

#     # If you want density weighting, replace mean with sum(density * B_parallel)/sum(density)

#     # --------------------------
#     # Toy spectral line parameters
#     # --------------------------
#     nv = 512
#     v = np.linspace(-5.0, 5.0, nv)   # velocity (or freq) axis in arbitrary units
#     sigma_v = 0.6                    # line width
#     I0 = 1.0                         # peak intensity

#     # weak-Zeeman proportionality constant (choose units so signal is visible)
#     # In real physics this depends on line rest freq and Landé g; here treat as tunable scalar
#     zeta = 0.2   # (velocity unit per unit B) -- tune to get visible splitting

#     # precompute base line (centered at v=0)
#     base_line = I0 * np.exp(-0.5 * (v/sigma_v)**2)

#     # allocate output cubes: (nx, ny, nv)
#     nx, ny = box_length, box_length
#     I_cube = np.zeros((nx, ny, nv))
#     V_cube = np.zeros((nx, ny, nv))

#     # loop over sightlines (vectorized approach possible; loop kept clear)
#     for i in range(nx):
#         for j in range(ny):
#             Bbar = B_los_map[i, j]                 # LOS field for that pixel
#             delta_v = zeta * Bbar                  # shift of sigma units
#             # Shifted right/left circular polarization profiles:
#             # RCP shifted to +delta, LCP shifted to -delta (or vice versa depending sign convention)
#             I_R = I0 * np.exp(-0.5 * ((v - delta_v)/sigma_v)**2)
#             I_L = I0 * np.exp(-0.5 * ((v + delta_v)/sigma_v)**2)
#             I = 0.5*(I_R + I_L)
#             V = I_R - I_L                          # circular pol. difference (Stokes V)
#             I_cube[i, j] = I
#             V_cube[i, j] = V

#     # --------------------------
#     # Add gaussian observational noise (optional)
#     # --------------------------
#     noise_rms = 5e-3
#     rng = np.random.default_rng(42)
#     V_cube += rng.normal(scale=noise_rms, size=V_cube.shape)
#     I_cube += rng.normal(scale=noise_rms, size=I_cube.shape)

#     # --------------------------
#     # Diagnostics: maps you might want
#     # --------------------------
#     # peak amplitude of V (signed) maximum circular polarization signal at each picel. Observable = 'Zeeman detection strength"
#     V_peak = np.max(V_cube, axis=2)   # shape (nx,ny)
#     # integrated absolute V (a simple proxy) Rough measure of total Polarized signal across line
#     V_int = np.trapezoid(np.abs(V_cube), x=v, axis=2)

#     # Plot B_los_map
#     fig, ax = plt.subplots(2,2, figsize=(8,8))
#     im0 = ax[0,0].imshow(B_los_map.T, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
#     ax[0,0].set_title(r'B$_{LOS}$ (z)')
#     ax[0,0].set_xlabel('x')
#     ax[0,0].set_ylabel('y')
#     plt.colorbar(im0, ax=ax[0,0])
#     # Plot V_peak
#     im1 = ax[1,0].imshow(V_peak.T, origin='lower')#, extent=[x.min(), x.max(), y.min(), y.max()])
#     ax[1,0].set_title('peak Stokes V')
#     ax[1,0].set_xlabel('x')
#     ax[1,0].set_ylabel('y')
#     plt.colorbar(im1, ax=ax[1,0])

#     # plot sample spectrum at center pixel
#     i0, j0 = nx//2, ny//2
#     ax[1,1].plot(v, I_cube[i0, j0], label='I')
#     ax[1,1].plot(v, V_cube[i0, j0], label='V (x10)')
#     ax[1,1].plot(v, 10*V_cube[i0, j0], label='10*V')  # scaled so visible
#     ax[1,1].legend()
#     ax[1,1].set_title(f'Spectra at pixel ({i0},{j0})')

#     # plot stokes U and Q Map
#     U = u[:,:,0]
#     Q = q[:,:,0]
#     COS = cos2g[:,:,0]
#     X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]), indexing='ij')
#     factor = 1 / np.max(COS)

#     phi = 0.5 * np.arctan2(U, Q)
#     x_scaled = np.sin(phi) * COS * factor
#     y_scaled = np.cos(phi) * COS * factor

#     ax[0,1].set_title('Stokes U and Q Map')
#     ax[0,1].quiver(X,Y,x_scaled,y_scaled, scale=2, scale_units='xy', headaxislength=0, headlength=0, headwidth=1, pivot='middle', color='blue', alpha=0.7)

#     ax[0,1].set_xlabel('X')
#     ax[0,1].set_ylabel('Y')

#     plt.tight_layout()
#     plt.show()

#     return U, Q, COS, B_los_map.T

# # u, q, cos, blos = zeeman_wvy(box_length=16, amplitude=1.2, frequency=2.0, y_const=0.5)
# # print(u, q, cos, blos)
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

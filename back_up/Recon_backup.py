# Status:
#   06/11/2025
#  
#   Currently transitioning 3D reconstruction to include sampling, cost, and refinement.
#   Majorly differentiating between 2.5D and 3D
#
# ************************************************************************
# Notes:
#
# 2.5D reconstruction verifies method, pure recon based off of equations
#
# 3D reconstruction employs sampling, cost function, and refinement
# 
# ************************************************************************
# Outline:
#   driver()
#       input: dim - dimensions (2 or 3)
#       output: 3D reconstructed field
#   dg.field()
#       Generates a 3D field of your choosen geometry 
#       options: uniform, wavy
#       output: Stokes U/Q 2D map to use as mock data
#   multigrid()
#       input: 2D mock data map, # dims
#       output: 3D reconstructed field
#       driver for entire code
#   rf.init_recon_2D() & rf.init_recon_3D()
#       input: 2D U/Q map
#       output: initial 3D reconstruction of U/Q, purely from geometry.
#       2D version reconstructs for essentiall a NxNx1 field, 
#       with 1 element in the z direction. 
#       3D version reconstructs for the full 3D field (NxNxN)
#   r_vcycle():
#       input: data 2D (u), restricted 2D level (u0), initial 3D reconstructed (u1), prolonged 3D (u3), dimensions (2.5 or 3)
#       output: saved reconstructed to u3_list
#       recursive function to restrict data and reconstruct at each level
#   restrict():
#       input: 2D U/Q map
#       output: restricted 2D U/Q map, next level down
#   p_vcycle():
#       input: data 3D (u1, initially reconstructed at that level),
#       guess 2D (u0, restricted map at that level), 
#       reconstructed list (u3, to be saved and returned),
#       J (ceiling), dim (dimensions, 2.5 or 3)
#       output: prolonged field
#       recursive function to prolong data at each level and return when done!
#   prolong_2D() & prolong_3D():
#       input: 2.5D or 3D map
#       output: prolonged field
#   rf.UQ_reconstruct()
#       input: bx, by, bz
#       output: finds stokes U/Q for each cell in the reconstructed 
#       field to compare to input and find original.
#   rf.residual()
#       input: U, Q, Ur, Qr
#       output: residual between input and reconstructed field
#   rf.visualize_3d_recon() & rf.visualize_25d_recon()
#       input: bx, by, bz, vector skip, vector length, name
#       output: 3D vector plot of reconstructed field
# ********************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import data_gen as dg
import recon_funcs as rf
#============================================================
# Drivers Below
#============================================================
#============================================================
#============================================================
# Prolong Cycle
#============================================================
def p_vcycle(datalist, iguess, uflist, J, dim): # u1_list is the list of initially reconstructed levels, u32 is the last level
    if len(datalist) != 0: 
        if iguess[0][0].shape[0] == 1: # this is for the first cell only, where u11 = u21, and skips sampling
            u21 = datalist.pop() # every time a level is used, remove last item in list
            uflist.append(u21) # save 1x1 square
            if dim == 2:
                u32 = rf.prolong_2D(u21)
            if dim == 3:
                u32 = rf.prolong_3D(u21)
            uflist.append(u32) # save 2x2 square
            p_vcycle(datalist, u32, uflist, J, dim)
        else:
            u12 = datalist.pop() # every time a level is used, remove
            # Check Level:
            if u12[0][0].shape[0] >= J: # last level; ceiling is hit. No more prolongation.
                # I think sampling goes here
                # input is u12(data), u32(guess), u22(sampled, best value)
                # then a cost function calculation
                # then some regularization? which could be in tandem with cost function, or sampling too!
                # u22 = regulate(data=u12,guess=iguess) # Add regulate later
                u22 = u12
                uflist.append(u22)
                return
            else: # still growing, still prolongating
                # u22 = regulate(data=u12,guess=iguess) # Add regulate later
                u22 = u12
                if dim == 2:               
                    u33 = rf.prolong_2D(u22)
                if dim == 3:
                    u33 = rf.prolong_3D(u22)
                uflist.append(u33)         
                p_vcycle(datalist, u33, uflist, J, dim)
    else:
        print('All levels reached, returning...') # once on last level, return

     
#============================================================
# Restriction cycle 
#============================================================
def r_vcycle(u, u0_list, u1_list, u3list, dim):
    
    if u[0][0].shape[0] == 1: # check size is 1 cell
        print('lvl == 1: Bottom reached; Begin Prolongation')
        J = u1_list[0][0].shape[0]
        p_vcycle(u1_list,u0_list[-1],u3list,J, dim) # start prolongation cycle at smallest cell

    else:
        if dim == 2:
            u0 = rf.restrict(u) # restrict to lower resolution
            u0_list.append(u0) # save each restricted 2D level
            u_1 = rf.init_recon_2D(u0) # initially reconstruct each level
        if dim == 3:
            u0 = rf.restrict(u) # restrict to lower resolution
            u0_list.append(u0) # save each restricted 2D level
            u_1 = rf.init_recon_3D(u0) # initially reconstruct each level
        u1_list.append(u_1) # save each 3D recon level
        r_vcycle(u0, u0_list, u1_list, u3list, dim) # restart cycle for next lvl
    
# ********************************************************************************
# u0 = initial observations, restricted: u0 = restrict(u)
# u1 = initially recosntructed/sampled, acts as data: u1 = reconstruct/sample(u0)
# u2 = reconstructed result from u1 data and u3 as guess: u2 = sample(u3|u1)
# u3 = prolonged data, starting point/initial guess: u3 = prolong(u2)
def multigrid(u, dim): # takes in 2D array of data, and number of runs for sampler
    J    = u[0][0].shape[0]
    print('Ceiling = ', J)
    if (J % 2 == 0):
        u0_list = [] # restricted levels
        u1_list = [] # initially reconstructed levels
        u3_list = []

        u0_list.append(u) # save first level
        if dim == 2:
            u1_list.append(rf.init_recon_2D(u))
        if dim == 3:
            u1_list.append(rf.init_recon_3D(u))

        # grab restricted lvls and initially recon lvls (u0,u1)
        r_vcycle(u, u0_list, u1_list, u3_list, dim) # u = data, u0 enters with saved data array, u1 enters empty.
        print('Multigrid Complete...')
    
    else:
        print('[multigrid]: J must be even: J=%4i' % (J))

    return u3_list        
# ******************************************************************************** 
def driver(dim): # driver creates fake data, which then gets sent to multigrid function to drive the whole production
    print('===============')
    print('Loading data...')

    # 2.5D reconstruction
    if dim == 2:
        print('Reconstructing in 2.5 Dimensions')
        # ------------------------------------------------------------------------------------
        # A single test:
        #------------------------------------------------------------------------------------
        # Generate Data:
        # UQpol_array                 = dg.uniform_field(bx0=0.5,by0=0.5,bz0=0,box_length=8,visual=1)
        # UQpol_array                 = dg.wave_3d(amplitude=1, frequency=1, box_length=8, z_contribution=0.5, visual=1, plotdex='xy')
        # UQpol_array                 = dg.toroidal_field_3D(box_length=8, visual=1, alpha=0, plotdex='xy')
        UQpol_array                 = dg.hourglass_3d(visual=0,nskip=1,n=8,n_theta=8,axis=0)
        #------------------------------------------------------------------------------------
        # Run Reconstruction:
        # ------------------------------------------------------------------------------------
        d3_field                    = multigrid(UQpol_array, dim)
        Ur, Qr, cos2gr, PAr, polr   = rf.UQ_reconstruct_2D(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][2])
        res                         = rf.residual(UQpol_array[0], UQpol_array[1], Ur, Qr)
        print(res)
        exit()
        ilabel = 'Hourglass Field'
        rf.visual_UQ(Ur,Qr,label=ilabel,plotdex='xy')
        rf.visualize_25d(d3_field[-1][0], d3_field[-1][1], d3_field[-1][2], name='Reconstructed 2.5D '+ilabel, plotdex='xy')
        rf.visualize_slice(d3_field[-1][0], d3_field[-1][1], label=ilabel, plotdex='xy')
        exit()
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # # Loop test:
        # ------------------------------------------------------------------------------------
        n = 100
        Bzlist    = np.linspace(-10,10,n)
        alpha     = np.linspace(0,np.pi,n)
        residuals = np.zeros(n)
        PAi_array = np.zeros(n)
        PAr_array = np.zeros(n)

        for i, iBz in enumerate(alpha):
            #------------------------------------------------------------------------------------
            # Generate Data:
            # UQpol_array                 = dg.uniform_field(bx0=0.4,by0=0.4,bz0=iBz,box_length=8,visual=0)
            # UQpol_array                 = dg.wave_3d(amplitude=1, frequency=1, box_length=8, z_contribution=iBz, visual=0, plotdex='xy')
            # UQpol_array                 = dg.toroidal_field_3D(box_length=8, visual=0, alpha=iBz, plotdex='xy')
            # Hourglass has set Z components, and are varying, so can't run this experiment on it.
            #------------------------------------------------------------------------------------
            d3_field                  = multigrid(UQpol_array, dim)
            Ur, Qr, cos2gr, PAr, polr = rf.UQ_reconstruct_2D(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][2])
            res                       = rf.residual(UQpol_array[0], UQpol_array[1], Ur, Qr)
            residuals[i] = res

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(alpha, residuals, marker='o', linestyle='none')
        ax.set_xlabel('Pitch Angle (rads)')
        ax.set_ylabel('Residual')
        fig.tight_layout()
        ax.grid()
        plt.show()

        # ------------------------------------------------------------------------------------
        print('==== done ====')
        return d3_field
        # ------------------------------------------------------------------------------------


    # This is for Bz components that vary along the line of sight. Don't need to touch this until 2.5D reconstruction is working and well.
    if dim == 3:
        print('Reconstructing in 3 Dimensions')
        UQpol_array               = dg.wave_3d(amplitude=1, frequency=1, box_length=16, z_contribution=0)
        d3_field                  = multigrid(UQpol_array, dim)
        Ur, Qr, cos2gr, PAr, polr = rf.UQ_reconstruct_3D(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][1])
        res                       = rf.residual(UQpol_array[0], UQpol_array[1], Ur, Qr)
        print(res)
        rf.visualize_3d_recon(d3_field[-1][0], d3_field[-1][1], d3_field[-1][2], nskip=2, name='Reconstructed 3D Wavy Field')
        print('==== done ====')
        return d3_field

# ********************************************************************************
# ********************************************************************************
field = driver(dim=3)




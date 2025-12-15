
# 2.5D Reconstruction
#
#   verifies method, pure recon based off of equations
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import data_gen as dg
import back_up.recon_funcs as rf

# ------------------------------------------------------------------------------------
def driver(): 
    print('Reconstructing in 2.5 Dimensions')
    # ------------------------------------------------------------------------------------
    # A single test:
    # ------------------------------------------------------------------------------------
    # Generate Data:
    ilabel = 'Output Uniform Field'
    # UQpol_array                 = dg.uniform_field(bx0=0.5,by0=0.5,bz0=0,box_length=8,visual=1)
    # UQpol_array                 = dg.wave_3d(amplitude=1, frequency=1, box_length=8, z_contribution=0.5, visual=1, plotdex='ij')
    UQpol_array                 = dg.toroidal_field_3D(box_length=8, visual=1, alpha=0, plotdex='ij')
    # UQpol_array                 = dg.hourglass_3d(visual=1,nskip=1,n=8,n_theta=8,axis=2)
    # ------------------------------------------------------------------------------------
    # Run Reconstruction for single test:
    # ------------------------------------------------------------------------------------
    d3_field                    = rf.init_recon_2D(UQpol_array)
    Ur, Qr, cos2gr, PAr, polr   = rf.UQ_reconstruct_2D(bx = d3_field[0], by = d3_field[1], bz = d3_field[2])
    res                         = rf.residual(UQpol_array[0], UQpol_array[1], Ur, Qr)
    print(res)
    # Visualize the output:
    
    rf.visual_UQ(Ur,Qr,label=ilabel,plotdex='ij')
    rf.visualize_slice(d3_field[0], d3_field[1], label=ilabel, plotdex='ij')

    # Below isn't working for wavy field test OR torodial... Not sure why, most likely indexing issue.
    # rf.visualize_25d(d3_field[0], d3_field[1], d3_field[2], name='Reconstructed 2.5D '+ilabel, plotdex='ij') 

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # # Loop test:
    # ------------------------------------------------------------------------------------
    # n = 100
    # Bzlist    = np.linspace(-10,10,n)
    # alpha     = np.linspace(0,np.pi,n)
    # residuals = np.zeros(n)
    # PAi_array = np.zeros(n)
    # PAr_array = np.zeros(n)

    # for i, iBz in enumerate(alpha):
    #     #------------------------------------------------------------------------------------
    #     # Generate Data:
    #     # UQpol_array                 = dg.uniform_field(bx0=0.4,by0=0.4,bz0=iBz,box_length=8,visual=0)
    #     # UQpol_array                 = dg.wave_3d(amplitude=1, frequency=1, box_length=8, z_contribution=iBz, visual=0, plotdex='xy')
    #     # UQpol_array                 = dg.toroidal_field_3D(box_length=8, visual=0, alpha=iBz, plotdex='xy')
    #     # Hourglass has set Z components, and are varying, so can't run this experiment on it.
    #     #------------------------------------------------------------------------------------
    #     d3_field                  = multigrid(UQpol_array)
    #     Ur, Qr, cos2gr, PAr, polr = rf.UQ_reconstruct_2D(bx = d3_field[-1][0], by = d3_field[-1][1], bz = d3_field[-1][2])
    #     res                       = rf.residual(UQpol_array[0], UQpol_array[1], Ur, Qr)
    #     residuals[i] = res

    # # plotting
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.plot(alpha, residuals, marker='o', linestyle='none')
    # ax.set_xlabel('Pitch Angle (rads)')
    # ax.set_ylabel('Residual')
    # fig.tight_layout()
    # ax.grid()
    # plt.show()

    # ------------------------------------------------------------------------------------
    print('==== done ====')
    return d3_field
    # ------------------------------------------------------------------------------------

# ********************************************************************************
# ********************************************************************************
field = driver()




import numpy as np
import matplotlib.pyplot as plt
import data_gen as dg
import back_up.recon_funcs as rf
import Sampler as sampler 

def zeeman_recon_3D(u): 
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
                print('BLOS size', BLOS.shape[0])
                if BLOS.shape[0] > 1:
                    nBLOS[i,j] = BLOS[iblos,jblos]
                else:
                    # print('BLOS value',BLOS[0][0])
                    nBLOS[i,j] = BLOS[0][0]
    
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

def p_vcycle(u1_list, b3_list, J, C): # u1_list restricted 2d DATA (U, Q, cos2g), b3_list is the prolonged layers, J is the level we're on

    if len(u1_list) != 0: 
        # this needs to change below
        if J == 1: # this is for the first cell only, where u11 = u21, and skips sampling
            
            u11 = u1_list.pop() # restricted layers, act as data
            
            b11 = zeeman_recon_3D(u11) # Reconstruct restricted 1x1 square into cube

            b31 = rf.prolong_3D(b11) # prolong into 2x2x2 cube
        
            b3_list.append(b31) # save 2x2x2 cube
            
            J = b31[0][0].shape[0]

            print('[checkpoint]: size of 1st cell recon:', np.shape(b11), 'size of 1st cell prolonged:', np.shape(b31))
            p_vcycle(u1_list, b3_list, J, C)

        else:            
            u12 = u1_list.pop() # restricted 2d DATA; every time a level is used, remove
            
            b12 = zeeman_recon_3D(u12) # reconstruct nxn square into nxnxn cube
            
            b32 = b3_list[-1] # grab previously prolonged cube
            
            print('[checkpoint]: B12 shape = ', np.shape(b12), 'B32 shape = ', np.shape(b32))

            if np.shape(b12) == np.shape(b32):
                # set parameters for sampling
                dtheta   = 0.4
                dphi     = 0.3
                R        = 12000
                rbins    = R//150
                ntrials  = 800
                nbins    = ntrials//80
                burn     = 500
                plotting = 0

                # Check Level:
                if u12[0][0].shape[0] < C:  # still growing, still prolongating

                    b22 = sampler.mcmc_driver(data=b12, prolonged=b32, dtheta=dtheta, dphi=dphi, rbins=rbins, R=R, 
                                              nbins=nbins, ntrials=ntrials, burn=burn, plotting=plotting)
                    print('[checkpoint]: Sampled B shape = ', np.shape(b22))

                    b33 = rf.prolong_3D(b22)

                    print('[checkpoint]: ','B22 shape = ', np.shape(b22), 'B33 shape = ', np.shape(b33))

                    J = b33[0][0].shape[0]

                    b3_list.append(b33)

                    p_vcycle(u1_list, b3_list, J, C)

                elif u12[0][0].shape[0] == C: # Hit last level. No more prolongation, but still want to sample 
                    print(u12[0][0].shape[0])
                    # Sample last level
                    b22 = sampler.mcmc_driver(data=b12, prolonged=b32, dtheta=dtheta, dphi=dphi, rbins=rbins, R=R, 
                                              nbins=nbins, ntrials=ntrials, burn=burn, plotting=plotting)
                    
                    b3_list.append(b22)

                    print('Ceiling Hit at J = ', u12[0][0].shape[0])
                    return
                else:
                    print('You have gone past input dimensions, something is wrong.')
            else:
                print('Reconstructed and Prolonged B field shapes do not match.')

    else:
        print('All levels reached, returning...') # once on last level, return


def r_vcycle(tdmap, u1_list, b3_list, C):
    J = tdmap[0][0].shape[0]
    if J == 1: # check size is 1 cell
        # print('[checkpoint]: check for list of restricted layers')
        print('lvl == 1: Bottom reached; Begin Prolongation')
        p_vcycle(u1_list, b3_list, J, C) # start prolongation cycle at smallest cell

    else:
        u0 = rf.restrict_zeeman_min(tdmap) # restrict to lower resolution
        print('shape:',np.shape(u0[0]), np.shape(u0[3]))
        u1_list.append(u0) # save each restricted 2D level
        r_vcycle(u0, u1_list, b3_list, C) # restart cycle for next lvl


# u0 = initial observations, restricted: u0 = restrict(u)
# u1 = initially reconstructed/sampled, acts as data: u1 = reconstruct/sample(u0)
# u2 = sampled result from u1 data and u3 as guess: u2 = sample(u3|u1)
# u3 = prolonged data, starting point/initial guess: u3 = prolong(u2)
def multigrid(u): # takes in 2D array of data, and number of runs for sampler
    C    = u[0][0].shape[0]
    print('Ceiling = ', C)
    if (C % 2 == 0):
        u1_list = [] # restricted
        b2_list = [] # sampled
        b3_list = [] # prolonged

        u1_list.append(u) # save first level

        # grab restricted lvls and initially recon lvls (u0,u1)
        r_vcycle(u, u1_list, b3_list, C) # u = data, u0 enters with saved data array, u1 enters empty.
        print('[Multigrid]: Complete.')
    
    else:
        print('[Multigrid]: J must be even: C=%4i' % (C))

    return b3_list    


print('===============')
print('Loading data...')
U, Q, COS, B_los_map, Bx, By, Bz = dg.wavy_zeeman_pol_min(box_length=8,amplitude=1.2,frequency=1,x_const=0.2,num_measurements=4,plotting=1)
UQpol_array = [U, Q, COS, B_los_map]

print('Reconstructing in 3 Dimensions')
d3_field = multigrid(UQpol_array)

bxr = d3_field[4][0]
byr = d3_field[4][1]
bzr = d3_field[4][2]

step = 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = np.meshgrid(np.arange(Bx.shape[0]),
                        np.arange(By.shape[0]),
                        np.arange(Bz.shape[0]),indexing='xy')
ax.quiver(X[::step, ::step, ::step], Y[::step, ::step, ::step], Z[::step, ::step, ::step], Bx[::step, ::step, ::step], 
          By[::step, ::step, ::step], Bz[::step, ::step, ::step], length=1.5, normalize=True, arrow_length_ratio=0, 
          color='dodgerblue')
ax.set_xlabel('Bx')
ax.set_ylabel('By')
ax.set_zlabel('Bz')
plt.show()


step = 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = np.meshgrid(np.arange(bxr.shape[0]),
                        np.arange(byr.shape[0]),
                        np.arange(bzr.shape[0]),indexing='xy')
ax.quiver(X[::step, ::step, ::step], Y[::step, ::step, ::step], Z[::step, ::step, ::step], bxr[::step, ::step, ::step], 
          byr[::step, ::step, ::step], bzr[::step, ::step, ::step], length=1.5, normalize=True, arrow_length_ratio=0, 
          color='dodgerblue')
ax.set_xlabel('Bx')
ax.set_ylabel('By')
ax.set_zlabel('Bz')
plt.show()
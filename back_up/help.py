import numpy as np
import matplotlib.pyplot as plt
from data_gen import uqpol_generator
from mpl_toolkits.mplot3d import Axes3D

########################################################
# x = np.linspace(0, 2 * np.pi, 16)
# y = np.linspace(0, 2 * np.pi, 16)
# z = np.linspace(0, 2 * np.pi, 16)
# X, Y, Z = np.meshgrid(x, y, z)

# # Generate single wave direction field
# angle = 1 * np.sin(1 * X)
# bx = np.cos(angle)
# by = np.sin(angle)
# bz = np.full_like(X, 0)

# cos2g         = (bx**2+by**2)/(bx**2+by**2+bz**2)
# q             = (by**2-bx**2)/(bx**2+by**2) * cos2g
# u             = 2*bx*by/(bx**2+by**2) * cos2g
# phi           = 0.5*np.arctan2(u,q)
# pol           = np.sqrt(u**2+q**2) # which is the same as cos2g in this approach

# plt.figure(figsize=(7,7))
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Input PA')
# for i in range(len(angle[0][0])):
#     for j in range(len(angle[0][1])):
#         iphi = angle[0][i][j]
#         x = 1 * np.cos(iphi)
#         y = 1 * np.sin(iphi)
#         plt.quiver(i, j, x, y, angles='xy', scale_units='xy', scale=1, headlength=0, headaxislength=0, headwidth=0)
# plt.show()
########################################################

def display_3d_vector_space(bx, by, bz, nskip=2, vlength=1, title='3D Vector Field'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of points
    X, Y, Z = np.meshgrid(np.arange(bx.shape[0])[::nskip],
                          np.arange(bx.shape[0])[::nskip],
                          np.arange(bx.shape[0])[::nskip])

    # Plot every nth vector
    ax.quiver(X, Y, Z, bx[::nskip, ::nskip, ::nskip], by[::nskip, ::nskip, ::nskip], 
              bz[::nskip, ::nskip, ::nskip], length=vlength, normalize=True, 
              arrow_length_ratio=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

# test prolongation and restriction;
# need to add 3rd dimension
def init_recon(u): 
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

                ibx      = np.sqrt(ibx2)
                iby      = isign*np.sqrt(iby2)
                ibz      = np.sqrt(ibz2)
                # print(ibx,iby,ibz)
                #save; 
                bx[n][i][j] = ibx
                by[n][i][j] = iby
                bz[n][i][j] = ibz

    return [bx,by,bz]
#============================================================
# Prolongation function
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

    i,j,k = np.meshgrid(np.arange(J,dtype=int),np.arange(J,dtype=int),np.arange(J,dtype=int)) 
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
# Restriction function
#============================================================
def restrict_3D(u):
    J     = u[0][0].shape[0]
    print('Restricting... Level = ', J)

    i,j,k   = np.meshgrid(np.arange(J//2),np.arange(J//2),np.arange(J//2))
    uc0    = 0.25*(u[0][2*i,2*j,2*k]+u[0][2*i,2*j+1,2*k]+u[0][2*i+1,2*j,2*k]+u[0][2*i+1,2*j+1,2*k]+
                   u[0][2*i,2*j,2*k+1]+u[0][2*i,2*j+1,2*k+1]+u[0][2*i+1,2*j,2*k+1]+u[0][2*i+1,2*j+1,2*k+1])
    uc1    = 0.25*(u[1][2*i,2*j,2*k]+u[1][2*i,2*j+1,2*k]+u[1][2*i+1,2*j,2*k]+u[1][2*i+1,2*j+1,2*k]+
                   u[1][2*i,2*j,2*k+1]+u[1][2*i,2*j+1,2*k+1]+u[1][2*i+1,2*j,2*k+1]+u[1][2*i+1,2*j+1,2*k+1])
    uc2    = 0.25*(u[2][2*i,2*j,2*k]+u[2][2*i,2*j+1,2*k]+u[2][2*i+1,2*j,2*k]+u[2][2*i+1,2*j+1,2*k]+
                   u[2][2*i,2*j,2*k+1]+u[2][2*i,2*j+1,2*k+1]+u[2][2*i+1,2*j,2*k+1]+u[2][2*i+1,2*j+1,2*k+1])
    uc     = [uc0,uc1,uc2]

    return uc


# A = np.zeros((3,4,4))
# A[0] = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
# A[1] = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
# A[2] = np.array([[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3]])
# print(A.shape[2])

UQarray = uqpol_generator(U0=0.6,Q0=-0.3,box_length=16)
field3d = init_recon(UQarray)
print(np.shape(field3d))
restricted_array = restrict_3D(field3d)
print(np.shape(restricted_array))
prolonged_array = prolong_3D(restricted_array)
print(np.shape(prolonged_array))

display_3d_vector_space(prolonged_array[0], prolonged_array[1], prolonged_array[2], nskip=2, vlength=1, title='3D Vector Field')


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


# def visualize_3d_vector(x, y, z):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Create a grid of points
#     X, Y, Z = np.meshgrid(np.arange(np.shape(x)[0]),
#                           np.arange(np.shape(x)[0]),
#                           np.arange(np.shape(x)[0]))

#     # Plot the vectors
#     ax.quiver(X, Y, Z, x, y, z, length=0.3, normalize=True)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Vector Field')

#     plt.show()

# # Example 2D array
# bx = [[0.70243222, 0.9365763 , 0.9365763 , 1.17072037],
#        [0.9365763 , 1.17072037, 0.70243222, 0.9365763 ],
#        [0.9365763 , 0.70243222, 1.17072037, 0.9365763 ],
#        [1.17072037, 0.9365763 , 0.9365763 , 0.70243222]]

# by = [[0.84896036, 0.9365763 , 0.9365763 , 1.02419224],
#        [0.9365763 , 1.02419224, 0.84896036, 0.9365763 ],
#        [0.9365763 , 0.84896036, 1.02419224, 0.9365763 ],
#        [1.02419224, 0.9365763 , 0.9365763 , 0.84896036]]

# bz = [[0.77567774, 0.9365763 , 0.9365763 , 1.09747486],
#        [0.9365763 , 1.09747486, 0.77567774, 0.9365763 ],
#        [0.9365763 , 0.77567774, 1.09747486, 0.9365763 ],
#        [1.09747486, 0.9365763 , 0.9365763 , 0.77567774]]

# # Example usage
# x = np.random.rand(4, 4, 4)
# y = np.random.rand(4, 4, 4)
# z = np.random.rand(4, 4, 4)

# visualize_3d_vector(bx, by, bz)

############################################

# bx0 = np.array([[0.53042442, 0.68814067, 0.44969414, 0.66962932],
#        [0.64081055, 0.44758132, 0.60960514, 0.5489666 ],
#        [0.51461842, 0.51575042, 0.63756548, 0.62310797],
#        [0.56464825, 0.64616262, 0.56172715, 0.53907772]])
 
# by0 = np.array([[0.48022472, 0.53230838, 0.41108595, 0.60419875],
#        [0.58851034, 0.66562088, 0.58167107, 0.57423016],
#        [0.54686563, 0.54495872, 0.55646907, 0.55451283],
#        [0.61303166, 0.52857892, 0.49813926, 0.58222434]]) 

# bz0 = np.array([[ 0.69859441, -0.49306208,  0.79295872, -0.43190327],
#        [ 0.49296798,  0.59717653, -0.53855393,  0.60736759],
#        [-0.66038312, -0.66107602,  0.53277803, -0.55159042],
#        [-0.55259798, -0.55052538,  0.66054514,  0.6086132 ]])

# Create subplots
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # Plot bx0
# axs[0].imshow(bx, cmap='viridis', interpolation='none')
# axs[0].set_title('bx0')

# # Plot by0
# axs[1].imshow(by, cmap='viridis', interpolation='none')
# axs[1].set_title('by0')

# # Plot bz0
# axs[2].imshow(bz, cmap='viridis', interpolation='none')
# axs[2].set_title('bz0')

# fig.colorbar(axs[0].imshow(bx, cmap='viridis', interpolation='none'), ax=axs[0])
# fig.colorbar(axs[1].imshow(by, cmap='viridis', interpolation='none'), ax=axs[1])
# fig.colorbar(axs[2].imshow(bz, cmap='viridis', interpolation='none'), ax=axs[2])

# # Display the plots
# plt.tight_layout()
# plt.show()


############################################

# list = [1,2,3,4,5,6,7,8,9,10]


# def popper(list):
#     if len(list) == 0:
#         print('empty')
#     else:
#         print(list)
#         x = list.pop()
#         popper(list)

# popper(list)


############################################

# def convolve_laplacian(array):
#     # Define a 2D Laplacian kernel
#     laplacian_kernel = np.array([[0, 1, 0],
#                                  [1, -4, 1],
#                                  [0, 1, 0]])
#     # Convolve the 2D array with the Laplacian kernel
#     # The mode='reflect' parameter handles the array boundaries by reflecting the array at the borders.
#     convolved_array = convolve(array, laplacian_kernel, mode='reflect')

#     return convolved_array


# array = np.random.rand(100, 100)  # Create a 100x100 array with random values

# convolved_array = convolve_laplacian(array)

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# # plot original array
# axs[0].imshow(array, cmap='viridis', interpolation='none')
# axs[0].set_title('Original Array')
# # Plot convolved array
# axs[1].imshow(convolved_array, cmap='viridis', interpolation='none')
# axs[1].set_title('Convolved Array with Laplacian Kernel')

# plt.tight_layout()
# plt.show()

############################################

# # testing initial reconstruction equations
# U = 0.5
# Q = 0.6

# # phi = 0.5*np.arctan(U/Q)
# # angle = np.linspace(0,180,36)
# # phi = np.pi*angle/180

# Bx2 = 1/(np.arctan2([phi],1) + 1)
# By2 = 1 - Bx2
# print(Bx2+By2)

# print(" ")
# # print('phi:', angle, 'Bx: ', np.sqrt(Bx2), 'By2: ', np.sqrt(By2))

# newphi = np.tan(np.sqrt(By2)/np.sqrt(Bx2))
# # print('new phi: ', newphi*180/(np.pi))
# newangle = newphi*180/(np.pi)

# plt.plot(angle, newangle, marker='o', linestyle = 'none')
# plt.xlabel('phi')
# plt.ylabel('reconstructed phi')
# plt.show()
# angle = U/Q
# print(np.arctan(angle))

############################################
U = 0.4
Q = 0.5


phi = 0.5*np.arctan(U/Q)
pol = np.sqrt(U**2 + Q**2)

phi = phi*180/np.pi

print(phi, pol)
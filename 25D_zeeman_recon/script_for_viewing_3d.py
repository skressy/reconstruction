import numpy as np
import matplotlib.pyplot as plt

# === USER-ADJUSTABLE PARAMETERS ===
# Grid size
g_size = 8
# Intensity and intrinsic polarization fraction
I0 = 1.0               # intensity per voxel (constant here)

# POS pattern parameters
frequency = 1.0        # frequency of the sinusoidal variation across the POS
# amplitude = 1.0        # amplitude of the sinusoidal variation across the POS
amplitude = np.linspace(0,1,g_size)

# magnetic field strength of Bx, By components:
Bx0 = 1.0              # POS B-field component along x (arbitrary units)
By0 = 1.0              # POS B-field component along y (arbitrary units)

# LOS (Zeeman) magnetic field (true value) and measurement noise
Bz_true = 0.5          # true LOS B-field (arbitrary units, e.g., microgauss) - constant here
zeeman_noise_sigma = 0.3  # Gaussian noise added to simulated Zeeman "measurement"

# Turbulence along LOS: controls angular perturbations (radians)
turb_sigma = 0.1       # larger -> stronger angular perturbations -> more depolarization
corr_r = 0.8         # AR(1) correlation coefficient along the z-axis (0 = white; close to 1 = highly correlated)

# DCF parameters (arbitrary units). For physical units, set rho and sigma_v appropriately.
rho = 1.0              # mass density (set to 1 for toy model)
sigma_v = 1.0          # LOS velocity dispersion (required by DCF formula)
cf_prefactor = np.sqrt(4 * np.pi * rho) * sigma_v


x = np.linspace(-np.pi, np.pi, g_size)
y = np.linspace(-np.pi, np.pi, g_size)
z = np.linspace(-np.pi, np.pi, g_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

Bx = np.zeros(X.shape)
By = np.zeros(Y.shape)
Bz = np.zeros(Z.shape)

Bz = np.ones_like(Z) * Bz_true

# Sinusoidal Field in POS
# Vary in amplitude along z-axis to test geometric depolarization
for i in range(len(Z)):
    theta0 = amplitude[i] * np.cos(frequency * X[:,:,i])

    Bx[:,:,i] = Bx0*np.cos(theta0)
    By[:,:,i] = By0*np.sin(theta0)


# fig, ax = plt.subplots(1, g_size, figsize=(16, 2))
# for i in range(g_size):
#     ax[i].quiver(
#         X[:,:,i], Y[:,:,i],
#         Bx[:,:,i], By[:,:,i],
#         pivot='middle', scale=2, scale_units='xy',headaxislength=0,
#         headlength=0,headwidth=1
#     )
#     ax[i].set_title(f"Slice {i+1}")

# plt.show()

print(Bx)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(X, Y, Z, Bx, By, Bz, length=0.5, normalize=True, 
        arrow_length_ratio=0, color='dodgerblue')
ax.set_xlabel('Bx')
ax.set_ylabel('By')
ax.set_zlabel('Bz')
ax.set_title('3D Magnetic Field with Geometric Depolarization')
plt.show()
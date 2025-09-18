import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, j0, j1, erfc

# === Parameters ===
n = 50          # resolution
n_theta = 20    # angular resolution
R_outer = 5.0
h = 1.0
B0 = 1.0
Bm = 1.0
betam = Bm / B0
N_terms = 3    # reduced to avoid overflow

# Grids (dimensionless: r, z ∈ [-2, 2])
r = np.linspace(-0.25, 0.25, n)
z = np.linspace(-0.25, 0.25, n)
R_grid, Z_grid = np.meshgrid(r, z, indexing='ij')

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

# === Plot Bz and Br components ===
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(Bz_array, origin='lower', cmap='viridis',
           extent=[z.min(), z.max(), r.min(), r.max()], aspect='auto')
plt.colorbar(label='Bz')
plt.title('Bz Component')
plt.xlabel('z')
plt.ylabel('r')

plt.subplot(1, 2, 2)
plt.imshow(Br_array, origin='lower', cmap='plasma',
           extent=[z.min(), z.max(), r.min(), r.max()], aspect='auto')
plt.colorbar(label='Br')
plt.title('Br Component')
plt.xlabel('z')
plt.ylabel('r')

plt.tight_layout()
plt.show()

# === 2D Streamplot for Debugging ===
plt.figure(figsize=(6,6))
plt.streamplot(z, r, Bz_array.T, Br_array.T, color='k', linewidth=1)
plt.xlabel('z')
plt.ylabel('r')
plt.title('Field Lines in r-z Plane')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


# === 3D Construction ===
r_skip = slice(None, None, 5)
z_skip = slice(None, None, 5)
r_sliced = r[r_skip]
z_sliced = z[z_skip]
Br_sliced = Br_array[r_skip, :][:, z_skip]
Bz_sliced = Bz_array[r_skip, :][:, z_skip]

theta = np.linspace(0, 2 * np.pi, n_theta)
R2D, T2D, Z3D = np.meshgrid(r_sliced, theta, z_sliced, indexing='ij')

X = R2D * np.cos(T2D)
Y = R2D * np.sin(T2D)

Br3D = np.repeat(Br_sliced[:, np.newaxis, :], n_theta, axis=1)
Bz3D = np.repeat(Bz_sliced[:, np.newaxis, :], n_theta, axis=1)

BX = Br3D * np.cos(T2D)
BY = Br3D * np.sin(T2D)
BZ = Bz3D

# === 3D Vector Field Plot ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.quiver(X, Y, Z3D, BX, BY, BZ, length=0.05, normalize=True,
          linewidth=1.5, arrow_length_ratio=0, color='dodgerblue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Magnetic Field: Hourglass Configuration')
plt.tight_layout()
plt.show()

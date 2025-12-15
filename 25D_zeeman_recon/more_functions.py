# MORE FUNCTIONS CAUSE I"M RUNNING OUT OF SPACE IN RECON_FUNCS.PY



############################################################
# NEAREST NON -1 FUNCTION
############################################################
def nearest_non_minus_one(grid, start, allow_diagonals=False):
    """
    grid: list of lists (rows x cols), values where -1 means "bad"
    start: (r, c) tuple, 0-based
    allow_diagonals: whether to consider 8 neighbors
    Returns: (r, c, dist) of nearest cell whose value != -1, or None if none found
    """
    from collections import deque

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    sr, sc = start
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("start out of bounds")

    # If start itself is valid
    if grid[sr][sc] != -1:
        return (sr, sc)

    # neighbor deltas
    if allow_diagonals:
        deltas = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        deltas = [(-1,0),(1,0),(0,-1),(0,1)]

    q = deque()
    q.append((sr, sc))
    visited = [[False]*cols for _ in range(rows)]
    visited[sr][sc] = True

    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                visited[nr][nc] = True
                if grid[nr][nc] != -1:
                    return (nr, nc) # d + 1 = distance
                q.append((nr, nc))

    return None

# # Example:
# grid = [
#     [-1, -1, -1, -1],
#     [-1,  5, -1, -1],
#     [-1, -1, -1,  2],
# ]
# print(nearest_non_minus_one(grid, (0,0)))   # -> (1,1,2)
# print(nearest_non_minus_one(grid, (2,3)))   # -> (2,3,0)

############################################################
# RECONSTRUCTION WITH ZEEMAN + POLARIZATION
############################################################
def zrecon(U, Q, BLOS, COS2G):
    import numpy as np

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

    return bx, by, bz


############################################################
# Reconstruction with POLARIZATION ONLY
############################################################
def precon(U, Q, COS2G, array_input):
    
    import numpy as np

    phi     = 0.5*np.arctan2(U,Q) # np.arctan takes radians
    sign    = np.sign(phi)
    by2     = 1/((np.tan(phi))**2 + 1)
    bx2     = 1 - by2
    bz2     = 1/COS2G * (bx2 + by2) * (1 - COS2G)

    bx = np.sqrt(bx2)
    by = np.sqrt(by2)
    bz = np.sqrt(bz2)

    # Check size of input data
    if array_input == True:
        # BY
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                iphi = phi[i,j]
                if iphi > np.pi/2 and iphi < 3*np.pi/2:
                    by[i,j] *= -1
                if iphi > -3*np.pi/2 and iphi < -np.pi/2:
                    by[i,j] *= -1
        # BX
                if iphi > np.pi and iphi < 2*np.pi:
                    bx[i,j] *= -1
                if iphi > -np.pi and iphi < 0:
                    bx[i,j] *= -1
    else:
        iphi = phi
        # BY
        if iphi > np.pi/2 and iphi < 3*np.pi/2:
            by *= -1
        if iphi > -3*np.pi/2 and iphi < -np.pi/2:
            by *= -1
        # BX
        if iphi > np.pi and iphi < 2*np.pi:
            bx *= -1
        if iphi > -np.pi and iphi < 0:
            bx *= -1

    return bx, by, bz
import numpy as np


def generate_3d_grid_matrix(k: int, seed: int = 42) -> np.ndarray:
    """
    Generuje macierz NxN (N = (2^k)^3), czyli N jest długością równą ilości kostek na ile byłby podzielony sześcian,
    jeżeli w każdej osi dokonamy 2^k cięć:
    - wiersz = wierzchołek
    - w kolumnach niezerowe losowe wartości dla sąsiadów (6 sąsiadów)
    - na przekątnej również wartość losowa (np. 10x większa)
    """
    dim = 2**k
    N = dim * dim * dim
    A = np.zeros((N, N), dtype=np.float64)

    rng = np.random.default_rng(seed)

    def idx(x: int, y: int, z: int) -> int:
        return x + y * dim + z * dim * dim

    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def rand_val() -> float:
        return float(rng.uniform(0.1, 1.0))

    for z in range(dim):
        for y in range(dim):
            for x in range(dim):
                row = idx(x, y, z)

                # diagonalna dominacja
                A[row, row] = 10.0 * rand_val()

                for dx, dy, dz in neigh:
                    neigh_x, neigh_y, neigh_z = x + dx, y + dy, z + dz
                    if 0 <= neigh_x < dim and 0 <= neigh_y < dim and 0 <= neigh_z < dim:
                        col = idx(neigh_x, neigh_y, neigh_z)
                        A[row, col] = rand_val()

    return A

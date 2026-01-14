import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # backend bez GUI (USTAW przed importem pyplot)

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from lab4.MatrixHierarchicalTree import MatrixHierarchicalTree
from lab4.MatrixHierarchicalTree import h_mv_mult, h_mult, _dense_block_from_node

# ===== ŚCIEŻKI =====
IMAGE_SAVE_PATH = r"lab4/processed_images/"
IMAGE_SAVE_PATH_REPORT = r"lab4/processed_images/report/"
IMAGE_READ_PATH = r"lab4/img.png"

# ===== UTWÓRZ KATALOGI =====
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
os.makedirs(IMAGE_SAVE_PATH_REPORT, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(IMAGE_SAVE_PATH_REPORT, f"zad4_{RUN_TS}")
os.makedirs(RESULTS_DIR, exist_ok=True)

TXT_PATH = os.path.join(RESULTS_DIR, "results.txt")


def generate_3d_grid_matrix(k: int, seed: int = 42) -> np.ndarray:
    """
    Generuje macierz NxN (N = (2^k)^3) opisującą topologię 3D siatki sześciennej:
    - wiersz = wierzchołek
    - w kolumnach niezerowe losowe wartości dla sąsiadów (6-neighborhood)
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
                r = idx(x, y, z)

                # diagonalna dominacja
                A[r, r] = 10.0 * rand_val()

                for dx, dy, dz in neigh:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < dim and 0 <= ny < dim and 0 <= nz < dim:
                        c = idx(nx, ny, nz)
                        A[r, c] = rand_val()

    return A


# ===== PARAMETRY KOMPRESJI =====
b_rank = 32
min_size = 4
delta = "mean"  # albo np. 1e-3, albo "median"

# ===== USTAWIENIA WALIDACJI MNOŻENIA =====
DO_MM = True  # ustaw False, jeśli chcesz pominąć mnożenie
MM_VALIDATE_UP_TO_K = (
    3  # dla k>3 walidacja A@A może być ciężka (u Ciebie i tak N rośnie)
)

# nagłówek TXT
with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write(f"ZAD 4 - wyniki | run={RUN_TS}\n")
    f.write(f"Parametry: b_rank={b_rank}, min_size={min_size}, delta={delta}\n")
    f.write(f"MM: DO_MM={DO_MM}, validate_up_to_k={MM_VALIDATE_UP_TO_K}\n\n")

# ===== URUCHOMIENIE =====
for k in [2, 3, 4]:  # możesz zmienić na [2, 3, 4]
    print(f"\n===== ZAD 4: k={k} =====")
    A = generate_3d_grid_matrix(k, seed=42)
    N = A.shape[0]
    print("N =", N)

    compressor = MatrixHierarchicalTree(
        delta=delta, b_rank=b_rank, min_size=min_size, delta_probe=256
    )

    root = compressor.create_tree(A)

    # --- RYSUNEK PODZIAŁU (partition) -> zapis do PNG ---
    canvas = compressor.draw_partition((N, N))
    fig = plt.figure()
    plt.title(f"Partition of compressed matrix (k={k}, N={N})")
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")

    part_path = os.path.join(RESULTS_DIR, f"partition_k{k}_N{N}.png")
    plt.savefig(part_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- MV: A_hat * x ---
    x = np.random.default_rng(0).standard_normal(N)
    y_h = h_mv_mult(root, x)

    # walidacja MV
    y_ref = A @ x
    err_mv = np.linalg.norm(y_ref - y_h) / (np.linalg.norm(y_ref) + 1e-15)
    print("MV relative error:", err_mv)

    # --- MM: A_hat * A_hat (hierarchical multiplication) ---
    err_mm = None
    mm_note = ""
    if DO_MM:
        print("MM (hierarchical) ...")
        C_root = h_mult(root, root, compressor)

        # walidacja MM: porównanie z A@A (gęste) tylko do pewnego k
        if k <= MM_VALIDATE_UP_TO_K:
            print("MM validate (dense A@A) ...")
            A2_ref = A @ A

            # rekonstrukcja wyniku hierarchicznego do macierzy gęstej
            A2_h = _dense_block_from_node(C_root)

            err_mm = np.linalg.norm(A2_ref - A2_h) / (np.linalg.norm(A2_ref) + 1e-15)
            print("MM relative error:", err_mm)
        else:
            mm_note = f"Pomijam walidację MM dla k={k} (k > {MM_VALIDATE_UP_TO_K})."
            print(mm_note)

    # --- dopisz do TXT ---
    with open(TXT_PATH, "a", encoding="utf-8") as f:
        f.write(f"k={k}, N={N}\n")
        f.write(f"err_mv={err_mv}\n")
        f.write(f"partition_png={part_path}\n")
        if DO_MM:
            if err_mm is not None:
                f.write(f"err_mm={err_mm}\n")
            else:
                f.write("err_mm=SKIPPED\n")
                if mm_note:
                    f.write(f"mm_note={mm_note}\n")
        f.write("\n")

print(f"\nZapisano obraz i wyniki TXT do: {RESULTS_DIR}")
print("TXT:", TXT_PATH)

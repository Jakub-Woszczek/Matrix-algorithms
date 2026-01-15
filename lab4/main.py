import os
import time
from datetime import datetime

import matplotlib

from lab4.matrix_generator import generate_3d_grid_matrix

matplotlib.use("Agg")  # backend bez GUI (USTAW przed importem pyplot)

import numpy as np
from matplotlib import pyplot as plt

from lab4.MatrixHierarchicalTree import MatrixHierarchicalTree
from lab4.MatrixHierarchicalTree import h_mv_mult, h_mult, _dense_block_from_node

IMAGE_SAVE_PATH = r"lab4/processed_images/"
IMAGE_SAVE_PATH_REPORT = r"lab4/processed_images/report/"
IMAGE_READ_PATH = r"lab4/img.png"

os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
os.makedirs(IMAGE_SAVE_PATH_REPORT, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(IMAGE_SAVE_PATH_REPORT, f"zad4_{RUN_TS}")
os.makedirs(RESULTS_DIR, exist_ok=True)

TXT_PATH = os.path.join(RESULTS_DIR, "results.txt")


b_rank = 32
min_size = 4
delta = "mean"


with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write(f"ZAD 4 - wyniki | run={RUN_TS}\n")
    f.write(f"Parametry: b_rank={b_rank}, min_size={min_size}, delta={delta}\n")

for k in [2, 3, 4]:
    print(f"\n----- ZAD 4: k={k} -----")
    A = generate_3d_grid_matrix(k, seed=42)
    N = A.shape[0]
    print("N =", N)

    compressor = MatrixHierarchicalTree(
        delta=delta, b_rank=b_rank, min_size=min_size, delta_probe=256
    )
    root = compressor.create_tree(A)

    # rysowanie podziału
    canvas = compressor.draw_partition((N, N))
    fig = plt.figure()
    plt.title(f"Partition of compressed matrix (k={k}, N={N})")
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")

    part_path = os.path.join(RESULTS_DIR, f"partition_k{k}_N{N}.png")
    plt.savefig(part_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # MV
    elapsed_time_vector = 0
    trial = 5
    for _ in range(trial):  # Dajmy 5 prób
        x = np.random.default_rng(0).standard_normal(N)
        start = time.perf_counter()
        y_h = h_mv_mult(root, x)
        end = time.perf_counter()
        elapsed_time_vector += end - start
    elapsed_time_vector /= trial

    # walidacja MV
    y_ref = A @ x
    err_mv = np.sum((y_ref - y_h) ** 2)
    err_mv_relative = np.linalg.norm(y_ref - y_h) ** 2 / np.linalg.norm(y_ref) ** 2
    print("MV error:", err_mv)
    print("MV time:", elapsed_time_vector)

    # MM
    err_mm = None
    print("MM (hierarchical) ...")
    elapsed_time_matrix = 0
    for _ in range(trial):
        start = time.perf_counter()
        C_root = h_mult(root, root, compressor)
        end = time.perf_counter()
        elapsed_time_matrix += end - start
    elapsed_time_matrix /= trial

    # walidacja MM
    print("MM validate (dense A@A) ...")
    A2_ref = A @ A
    # rekonstrukcja wyniku hierarchicznego do macierzy gęstej
    A2_h = _dense_block_from_node(C_root)
    err_mm = np.sum((A2_h - A2_ref) ** 2)
    err_mm_relative = np.linalg.norm(A2_h - A2_ref) ** 2 / np.linalg.norm(A2_ref) ** 2
    print("MM relative error:", err_mm)
    print("MM time:", elapsed_time_matrix)

    # zapis do TXT
    with open(TXT_PATH, "a", encoding="utf-8") as f:
        f.write(f"k={k}, N={N}\n")
        f.write(f"err_mv={err_mv}\n")
        f.write(f"err_mv_relative={err_mv_relative}\n")
        f.write(f"time_mv={elapsed_time_vector}\n")
        f.write(f"partition_png={part_path}\n")
        if err_mm is not None:
            f.write(f"err_mm={err_mm}\n")
            f.write(f"err_mm_relative={err_mm_relative}\n")
            f.write(f"elapsed_time_matrix={elapsed_time_matrix}\n")
        else:
            f.write("err_mm=SKIPPED\n")
        f.write("\n")

print(f"\nZapisano obraz i wyniki TXT do: {RESULTS_DIR}")
print("TXT:", TXT_PATH)

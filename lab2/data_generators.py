import csv
import time
import tracemalloc
import numpy as np
from pathlib import Path

from lab2.inversion import recursion_inversion
from lab2.Gauss import recursive_gauss_elimination
from lab2.LU import lu_recursive

from lab1.binet import binet_multiplication
from lab1.strassen import strassen_multiplication


def generate_lu(csv_save_path="lab2/data/", method="Binet"):
    """
    Generuje dane pomiarowe dla rekurencyjnej faktoryzacji LU
    z użyciem wybranego algorytmu mnożenia (Binet lub Strassen).
    Zapisuje do CSV: średni czas, średnie zużycie pamięci (peak) i średnie FLOPs.
    """
    repeat_times = 1
    a, b, step = 5, 80, 2
    sizes = [i for i in range(a, b, step)]

    if method == "Binet":
        mult_function = binet_multiplication
    elif method == "Strassen":
        mult_function = strassen_multiplication
    else:
        raise ValueError("Unknown method type. Use 'Binet' or 'Strassen'.")

    Path(csv_save_path).mkdir(parents=True, exist_ok=True)
    filename = f"{method}_LU_start-{a}_end-{b}_step-{step}_repeat-{repeat_times}.csv"
    full_path = Path(csv_save_path) / filename

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "average_time", "average_memory", "average_flops"])

        for n in sizes:
            total_time = 0.0
            total_memory = 0.0
            total_flops = 0

            print(f"⚙️ Running {method} LU for size {n}x{n}")

            for rep in range(repeat_times):
                print(f"  🔁 Repeat {rep + 1}/{repeat_times}")

                # losowa, lekko dodiagonalizowana macierz (by uniknąć zerowych pivotów)
                A = np.random.rand(n, n) + np.eye(n) * 0.5

                tracemalloc.start()
                start = time.time()

                try:
                    # >>> właściwe LU, nie inwersja <<<
                    L, U, flops = lu_recursive(A, recursion_inversion, mult_function)
                except Exception as e:
                    print(f"❌ Error at size {n}: {e}")
                    tracemalloc.stop()
                    continue

                current, peak = tracemalloc.get_traced_memory()
                end = time.time()
                tracemalloc.stop()

                total_time += end - start
                total_memory += peak
                total_flops += flops

                # (opcjonalnie) szybka weryfikacja rekonstrukcji — nie wpływa na FLOPs
                # rel_err = np.linalg.norm(A - L @ U, ord='fro') / (np.linalg.norm(A, ord='fro') + 1e-15)
                # print(f"    recon rel_err={rel_err:.2e}")

            avg_time = total_time / repeat_times
            avg_memory = total_memory / repeat_times
            avg_flops = total_flops / repeat_times

            writer.writerow([n, avg_time, avg_memory, avg_flops])
            print(
                f"✅ size={n} | time={avg_time:.4f}s | mem={avg_memory/1e6:.2f}MB | flops={avg_flops}"
            )

    print(f"\n📊 Zapisano wyniki do pliku: {full_path}")


def generate_inv(csv_save_path="lab2/data/", method="Binet"):
    """
    Generuje dane pomiarowe dla odwracania macierzy (inverse)
    z użyciem wybranego algorytmu mnożenia (Binet lub Strassen).
    """
    repeat_times = 1
    a, b, step = 5, 80, 2
    sizes = [i for i in range(a, b, step)]

    if method == "Binet":
        mult_function = binet_multiplication
    elif method == "Strassen":
        mult_function = strassen_multiplication
    else:
        raise ValueError("Unknown method type. Use 'Binet' or 'Strassen'.")

    Path(csv_save_path).mkdir(parents=True, exist_ok=True)
    filename = (
        f"{method}_inverse_start-{a}_end-{b}_step-{step}_repeat-{repeat_times}.csv"
    )
    full_path = Path(csv_save_path) / filename

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "average_time", "average_memory", "average_flops"])

        for n in sizes:
            total_time = 0.0
            total_memory = 0.0
            total_flops = 0

            print(f"⚙️ Running {method} inversion for size {n}x{n}")

            for rep in range(repeat_times):
                print(f"  🔁 Repeat {rep + 1}/{repeat_times}")

                A = np.random.rand(n, n) + np.eye(n) * 0.5

                tracemalloc.start()
                start = time.time()

                try:
                    A_inv, flops = recursion_inversion(A, mult_function)
                except Exception as e:
                    print(f"❌ Error at size {n}: {e}")
                    tracemalloc.stop()
                    continue

                current, peak = tracemalloc.get_traced_memory()
                end = time.time()
                tracemalloc.stop()

                total_time += end - start
                total_memory += peak
                total_flops += flops

            avg_time = total_time / repeat_times
            avg_memory = total_memory / repeat_times
            avg_flops = total_flops / repeat_times

            writer.writerow([n, avg_time, avg_memory, avg_flops])
            print(
                f"✅ size={n} | time={avg_time:.4f}s | mem={avg_memory/1e6:.2f}MB | flops={avg_flops}"
            )

    print(f"\n📊 Zapisano wyniki do pliku: {full_path}")


def generate_gauss(csv_save_path="lab2/data/", method="Binet"):
    """
    Generuje dane pomiarowe dla rekurencyjnej eliminacji Gaussa (z PDF),
    korzystając z wybranego algorytmu mnożenia macierzy (Binet lub Strassen).

    Dane:
      - średni czas wykonania (s)
      - średnie zużycie pamięci (B)
      - średnia liczba FLOPs
    """

    repeat_times = 1
    a, b, step = 5, 80, 2
    sizes = [i for i in range(a, b, step)]

    if method == "Binet":
        mult_function = binet_multiplication
    elif method == "Strassen":
        mult_function = strassen_multiplication
    else:
        raise ValueError("Unknown method type. Use 'Binet' or 'Strassen'.")

    Path(csv_save_path).mkdir(parents=True, exist_ok=True)
    filename = f"{method}_Gauss_start-{a}_end-{b}_step-{step}_repeat-{repeat_times}.csv"
    full_path = Path(csv_save_path) / filename

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "average_time", "average_memory", "average_flops"])

        for n in sizes:
            total_time = 0.0
            total_memory = 0.0
            total_flops = 0

            print(f"⚙️ Running {method} Gauss elimination for size {n}x{n}")

            for rep in range(repeat_times):
                print(f"  🔁 Repeat {rep + 1}/{repeat_times}")

                # losowa dobrze uwarunkowana macierz
                A = np.random.rand(n, n) + np.eye(n) * 0.5
                b = np.random.rand(n, 1)

                tracemalloc.start()
                start = time.time()

                try:
                    _, _, flops = recursive_gauss_elimination(
                        A, recursion_inversion, mult_function, lu_recursive, b
                    )
                except Exception as e:
                    print(f"❌ Error at size {n}: {e}")
                    tracemalloc.stop()
                    continue

                current, peak = tracemalloc.get_traced_memory()
                end = time.time()
                tracemalloc.stop()

                total_time += end - start
                total_memory += peak
                total_flops += flops

            avg_time = total_time / repeat_times
            avg_memory = total_memory / repeat_times
            avg_flops = total_flops / repeat_times

            writer.writerow([n, avg_time, avg_memory, avg_flops])
            print(
                f"✅ size={n} | time={avg_time:.4f}s | mem={avg_memory/1e6:.2f}MB | flops={avg_flops}"
            )

    print(f"\n📊 Zapisano wyniki do pliku: {full_path}")

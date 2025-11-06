import csv
import time
import tracemalloc
import numpy as np
from pathlib import Path

from lab2.inversion import recursion_inversion
from lab1.binet import binet_multiplication
from lab1.strassen import strassen_multiplication



def generate_lu(csv_save_path="lab3/data/", method="Binet"):
    """
    Generuje dane pomiarowe dla faktoryzacji LU (rekurencyjnej)
    z użyciem wybranego algorytmu mnożenia i odwracania.

    method: "Binet" lub "Strassen"
    """
    repeat_times = 5
    a, b, step = 5, 30, 2
    sizes = [i for i in range(a, b, step)]

    # wybór metody mnożenia
    if method == "Binet":
        mult_function = binet_multiplication
    elif method == "Strassen":
        mult_function = strassen_multiplication
    else:
        raise ValueError("Unknown method type. Use 'Binet' or 'Strassen'.")

    # adapter do odwracania (korzysta z tego samego algorytmu mnożenia)
    def inversion_adapter(M):
        return recursion_inversion(M, mult_function)

    # przygotowanie ścieżki i pliku
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

            print(f"⚙️ Running {method} LU for size {n}×{n}")

            for rep in range(repeat_times):
                print(f"  🔁 Repeat {rep + 1}/{repeat_times}")

                A = np.random.rand(n, n) + np.eye(n) * 0.5  # stabilna macierz

                tracemalloc.start()
                start = time.time()

                try:
                    L, U, flops = lu_recursive_with_ops(A, mult_function, inversion_adapter)
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

            # uśrednianie
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
    repeat_times = 5
    a, b, step = 5, 10, 2
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

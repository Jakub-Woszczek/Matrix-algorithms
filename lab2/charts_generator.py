import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# (możesz zostawić swoją wersję create_chart_multiple; tu jest z katalogiem lab2/charts)
def create_chart_multiple(
    datasets, x_description=None, y_description=None, title=None, out_dir="lab2/charts"
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{title}.png"
    plt.figure(figsize=(8, 6))

    for x_vals, y_vals, label in datasets:
        plt.plot(x_vals, y_vals, linestyle="-", label=label)

    if x_description:
        plt.xlabel(x_description)
    if y_description:
        plt.ylabel(y_description)

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"✅ saved: {path}")


def _read_csv(path):
    """Helper: wczytuje CSV do dict."""
    mb = 1024**2
    data = {"size": [], "average_time": [], "average_memory": [], "average_flops": []}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "size":
                continue
            data["size"].append(float(row[0]))
            data["average_time"].append(float(row[1]))
            data["average_memory"].append(float(row[2]) / mb)
            data["average_flops"].append(float(row[3]))
    return data


def generate_inv(
    data_dir="lab2/data",
    charts_dir="lab2/charts",
    filename_pattern="{method}_inverse_start-5_end-50_step-1_repeat-5.csv",
    methods=("Binet", "Strassen"),
):

    # wczytaj dane dla wszystkich metod
    series = {}
    for m in methods:
        path = Path(data_dir) / filename_pattern.format(method=m)
        series[m] = _read_csv(path)

    # ===================== TIME (z n³) =====================
    datasets = [(series[m]["size"], series[m]["average_time"], m) for m in methods]

    first_key = next(iter(series.keys()))
    base_sizes = np.asarray(series[first_key]["size"], dtype=float)
    n3 = base_sizes**3

    # dopasowanie skali n³ do maksymalnej wartości czasu
    all_times = np.concatenate([series[m]["average_time"] for m in methods])
    scale = np.max(all_times) / np.max(n3)
    n3_scaled = n3 * scale

    # dodaj serię teoretyczną
    datasets.append((base_sizes, n3_scaled, "n³ (theoretical)"))

    create_chart_multiple(
        datasets,
        "Size",
        "Computing time [s]",
        "Gauss - Computing Time Comparison (with theoretical n³)",
        charts_dir,
    )

    # Memory
    datasets = [(series[m]["size"], series[m]["average_memory"], m) for m in methods]
    create_chart_multiple(
        datasets,
        "Size",
        "Memory usage [MB]",
        "Inverse - Memory Usage Comparison",
        charts_dir,
    )

    # FLOPs
    datasets = [(series[m]["size"], series[m]["average_flops"], m) for m in methods]
    create_chart_multiple(
        datasets, "Size", "FLOPs", "Inverse - FLOPs Comparison", charts_dir
    )


def generate_lu(
    data_dir="lab2/data",
    charts_dir="lab2/charts",
    filename_pattern="{method}_LU_start-5_end-10_step-2_repeat-5.csv",
    methods=("Binet", "Strassen"),
):
    """
    Generuje 3 wykresy porównawcze (czas, pamięć, FLOPs) dla INVERSE w lab2.
    Czyta pliki CSV:
        lab2/data/Binet_inverse_start-5_end-200_step-2_repeat-5.csv
        lab2/data/Strassen_inverse_start-5_end-200_step-2_repeat-5.csv
    """
    # wczytaj dane dla wszystkich metod
    series = {}
    for m in methods:
        path = Path(data_dir) / filename_pattern.format(method=m)
        series[m] = _read_csv(path)

    # ===================== TIME (z n³) =====================
    datasets = [(series[m]["size"], series[m]["average_time"], m) for m in methods]

    first_key = next(iter(series.keys()))
    base_sizes = np.asarray(series[first_key]["size"], dtype=float)
    n3 = base_sizes**3

    # dopasowanie skali n³ do maksymalnej wartości czasu
    all_times = np.concatenate([series[m]["average_time"] for m in methods])
    scale = np.max(all_times) / np.max(n3)
    n3_scaled = n3 * scale

    # dodaj serię teoretyczną
    datasets.append((base_sizes, n3_scaled, "n³ (theoretical)"))

    create_chart_multiple(
        datasets,
        "Size",
        "Computing time [s]",
        "Gauss - Computing Time Comparison (with theoretical n³)",
        charts_dir,
    )

    # Memory
    datasets = [(series[m]["size"], series[m]["average_memory"], m) for m in methods]
    create_chart_multiple(
        datasets,
        "Size",
        "Memory usage [MB]",
        "LU - Memory Usage Comparison",
        charts_dir,
    )

    # FLOPs
    datasets = [(series[m]["size"], series[m]["average_flops"], m) for m in methods]
    create_chart_multiple(
        datasets, "Size", "FLOPs", "LU - FLOPs Comparison", charts_dir
    )


def generate_gauss(
    data_dir="lab2/data",
    charts_dir="lab2/charts",
    filename_pattern="{method}_Gauss_start-5_end-30_step-5_repeat-3.csv",
    methods=("Binet", "Strassen"),
):
    """
    Generuje wykresy dla rekurencyjnej eliminacji Gaussa:
      - czas (z dodaną teoretyczną serią ~ n³)
      - pamięć
      - FLOPs
    """

    # Wczytanie danych z CSV
    series = {}
    for m in methods:
        path = Path(data_dir) / filename_pattern.format(method=m)
        s = _read_csv(path)
        # konwersja na numpy arrays
        series[m] = {
            "size": np.asarray(s["size"], dtype=float),
            "average_time": np.asarray(s["average_time"], dtype=float),
            "average_memory": np.asarray(s["average_memory"], dtype=float),
            "average_flops": np.asarray(s["average_flops"], dtype=float),
        }

    # Wspólna lista rozmiarów z pierwszej serii
    first_key = next(iter(series.keys()))
    base_sizes = np.asarray(series[first_key]["size"], dtype=float)
    n3 = base_sizes**3

    # ===================== TIME (z n³) =====================
    datasets = [(series[m]["size"], series[m]["average_time"], m) for m in methods]

    # dopasowanie skali n³ do maksymalnej wartości czasu
    all_times = np.concatenate([series[m]["average_time"] for m in methods])
    scale = np.max(all_times) / np.max(n3)
    n3_scaled = n3 * scale

    # dodaj serię teoretyczną
    datasets.append((base_sizes, n3_scaled, "n³ (theoretical)"))

    create_chart_multiple(
        datasets,
        "Size",
        "Computing time [s]",
        "Gauss - Computing Time Comparison (with theoretical n³)",
        charts_dir,
    )

    # ===================== MEMORY =====================
    datasets = [(series[m]["size"], series[m]["average_memory"], m) for m in methods]
    create_chart_multiple(
        datasets,
        "Size",
        "Memory usage [MB]",
        "Gauss - Memory Usage Comparison",
        charts_dir,
    )

    # ===================== FLOPs =====================
    datasets = [(series[m]["size"], series[m]["average_flops"], m) for m in methods]
    create_chart_multiple(
        datasets,
        "Size",
        "FLOPs",
        "Gauss - FLOPs Comparison",
        charts_dir,
    )

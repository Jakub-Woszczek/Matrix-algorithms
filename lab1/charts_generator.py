import csv
import matplotlib.pyplot as plt


def create_chart_multiple(datasets, x_description=None, y_description=None, title=None):
    """
    Create a chart with multiple datasets.

    Parameters:
    - datasets: list of tuples (x_vals, y_vals, label)
        Example: [([1,2,3], [4,5,6], "Series 1"), ([2,3,4], [10,15,20], "Series 2")]
    - x_description: label for the x-axis
    - y_description: label for the y-axis
    - name: filename to save chart as PNG
    """
    path = "lab1/charts/" + title
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


def generate_chart():
    binet_csv_path = r"lab1/data/Binet_start-5_end-200_step-2_repeat-5.csv"
    strassen_csv_path = "lab1/data/Strassen_start-5_end-200_step-2_repeat-5.csv"

    binet_data = {
        "size": [],
        "average_time": [],
        "average_memory": [],
        "average_flops": [],
    }
    strassen_data = {
        "size": [],
        "average_time": [],
        "average_memory": [],
        "average_flops": [],
    }
    config = {}
    # MB factor
    mb_factor = 1024**2
    with open(binet_csv_path, "r", newline="") as binet_f, open(
        strassen_csv_path, "r", newline=""
    ) as strassen_f:

        binet_reader = csv.reader(binet_f, delimiter=",")
        strassen_reader = csv.reader(strassen_f, delimiter=",")

        for row in binet_reader:
            if not row:
                continue
            if row[0].startswith("#"):
                # Optional, in 'save_data' i have this section commented
                if "=" in row[0]:
                    key, value = row[0][1:].split("=", 1)
                    config[key.strip()] = value.strip()
                continue
            if row[0] == "size":
                continue

            binet_data["size"].append(float(row[0]))
            binet_data["average_time"].append(float(row[1]))
            binet_data["average_memory"].append(float(row[2]) / mb_factor)
            binet_data["average_flops"].append(float(row[3]))

        for row in strassen_reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if row[0] == "size":
                continue

            strassen_data["size"].append(float(row[0]))
            strassen_data["average_time"].append(float(row[1]))
            strassen_data["average_memory"].append(float(row[2]) / mb_factor)
            strassen_data["average_flops"].append(float(row[3]))

    # Times
    binet_time_dataset = (binet_data["size"], binet_data["average_time"], "Binet")
    strassen_time_dataset = (
        strassen_data["size"],
        strassen_data["average_time"],
        "Strassen",
    )
    create_chart_multiple(
        [binet_time_dataset, strassen_time_dataset],
        "Size",
        "Computing time",
        "Computing Time Comparison",
    )

    # Memory
    binet_memory_dataset = (binet_data["size"], binet_data["average_memory"], "Binet")
    strassen_memory_dataset = (
        strassen_data["size"],
        strassen_data["average_memory"],
        "Strassen",
    )
    create_chart_multiple(
        [binet_memory_dataset, strassen_memory_dataset],
        "Size",
        "Memory usage [MB]",
        "Memory Usage Comparison",
    )

    # FLOPs
    binet_flops_dataset = (binet_data["size"], binet_data["average_flops"], "Binet")
    strassen_flops_dataset = (
        strassen_data["size"],
        strassen_data["average_flops"],
        "Strassen",
    )
    create_chart_multiple(
        [binet_flops_dataset, strassen_flops_dataset],
        "Size",
        "FLOPs",
        "FLOPs Comparison",
    )

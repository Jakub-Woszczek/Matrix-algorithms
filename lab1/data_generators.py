import csv
import time
import tracemalloc
import numpy as np


def generate_computing_time(multiplication_function, function=None, csv_save_path=""):
    repeat_times = 5
    a, b, step = 5, 200, 2

    if function == "Binet":
        sizes = [i for i in range(a, b, step)]  # Binet
    elif function == "Strassen":
        sizes = [i for i in range(a, b, step)]  # Strassen
    else:
        raise Exception("Unknown function type")

    filename = f"{function}_start-{a}_end-{b}_step-{step}_repeat-{repeat_times}.csv"
    full_path = csv_save_path + filename

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Optional: write config as comments at top
        # f.write(f"# function_name={function}\n")
        # f.write(f"# multiplication_function={multiplication_function.__name__}\n")
        # f.write(f"# repeat_times={repeat_times}\n")
        # f.write(f"# start_val={a}\n")
        # f.write(f"# end_val={b}\n")
        generate_computing_time  # f.write(f"# step={step}\n")
        writer.writerow(["size", "average_time", "average_memory", "average_flops"])

        for n in sizes:
            total_time = 0.0
            total_memory = 0.0
            total_flops = 0
            print(f"Running {multiplication_function.__name__} for {n} size")

            for rep in range(repeat_times):
                print(f"Repeat {rep}")
                A = np.random.rand(n, n)
                B = np.random.rand(n, n)

                tracemalloc.start()
                start = time.time()
                _, flops = multiplication_function(A, B)
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
            print(f"Avg time {avg_time}")

    return None

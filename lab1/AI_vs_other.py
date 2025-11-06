import numpy as np
import time
from lab1.binet import binet_multiplication
from AI_algo import new_AI


def measure_time(func, A, B):
    start = time.perf_counter()
    func(A, B)
    return time.perf_counter() - start


scales = [i for i in range(2, 4)]

print("{:<8} {:<15} {:<15}".format("k", "Binet [s]", "AI [s]"))

print("-" * 55)

for k in scales:
    rows_A = 4**k
    cols_A = 5**k
    rows_B = 5**k
    cols_B = 5**k

    A = np.random.randint(0, 10, (rows_A, cols_A))
    B = np.random.randint(0, 10, (rows_B, cols_B))

    t_binet = measure_time(binet_multiplication, A, B)
    t_ai = measure_time(new_AI, A, B)

    print("{:<8} {:<15.6f} {:<15.6f}".format(k, t_binet, t_ai))

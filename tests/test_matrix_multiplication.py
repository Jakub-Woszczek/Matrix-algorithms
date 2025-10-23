import numpy as np


def test_matrix_multiplication_fn(multiplication_function):
    """Test the correctness of a matrix multiplication algorithm"""
    sizes = [i for i in range(3, 100, 7)]

    for n in sizes:
        print(f"\n=== Test for {n}x{n} matrix ===")

        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        C_binet = multiplication_function(A, B)
        C_np = np.dot(A, B)

        if np.allclose(C_binet, C_np):
            print("✅ Correct result (matches numpy.dot)")
        else:
            print("❌ Incorrect result (does not match numpy.dot!)")

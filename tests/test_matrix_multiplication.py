import numpy as np


def test_matrix_multiplication_fn(multiplication_function):
    """Test the correctness of a matrix multiplication algorithm"""
    # sizes = [i for i in range(2, 20)]
    sizes = [i for i in range(5, 30, 1)]
    print(f"testing {multiplication_function.__name__}")

    for n in sizes:
        print(f"\n=== Test for {n}x{n} matrix ===")

        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        C, flops = multiplication_function(A, B)
        C_np = np.dot(A, B)

        if np.allclose(C, C_np):
            print("✅ Correct result (matches numpy.dot)")
            print(f"Flops: {flops}")
        else:
            print("❌ Incorrect result (does not match numpy.dot!)")

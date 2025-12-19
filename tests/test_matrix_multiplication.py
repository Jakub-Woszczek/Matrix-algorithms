import numpy as np

rng = np.random.default_rng()  # generator


def test_matrix_multiplication_fn(multiplication_function):
    """Test the correctness of a matrix multiplication algorithm"""
    # sizes = [i for i in range(2, 20)]
    sizes = [i for i in range(5, 30, 1)]
    print(f"testing {multiplication_function.__name__}")

    for n in sizes:
        print(f"\n=== Test for {n}x{n} matrix ===")

        rand = rng.integers(1, 9)
        A = np.random.rand(n + rng.integers(1, 30), n + rand)  # zakres [a, b], n+1)
        B = np.random.rand(n + rand, n)

        C, flops = multiplication_function(A, B)
        C_np = np.dot(A, B)

        if np.allclose(C, C_np):
            print("✅ Correct result (matches numpy.dot)")
            print(f"Flops: {flops}")
        else:
            print("❌ Incorrect result (does not match numpy.dot!)")

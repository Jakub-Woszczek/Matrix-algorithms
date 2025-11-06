import numpy as np
import time

def test_matrix_inversion_fn(inversion_function, multiplication_function):
    """
    Test the correctness and performance of a matrix inversion algorithm
    using a user-provided matrix multiplication function.
    """

    sizes = [i for i in range(3, 20, 2)]
    print(f"testing {inversion_function.__name__} with {multiplication_function.__name__}")

    for n in sizes:
        print(f"\n=== Test for {n}x{n} matrix ===")

        # Losowa macierz z lekkim dodaniem diagonali, by uniknąć macierzy osobliwych
        A = np.random.rand(n, n) + np.eye(n) * 0.5
        print(type(A))

        try:
            start_time = time.perf_counter()
            A_inv, flops = inversion_function(A, multiplication_function)
            elapsed = time.perf_counter() - start_time
        except Exception as e:
            print(f"❌ Exception during inversion: {e}")
            continue

        # Sprawdź poprawność: A * A⁻¹ ≈ I
        I_approx = np.dot(A, A_inv)
        I_true = np.eye(n)
        error_norm = np.linalg.norm(I_approx - I_true)

        if np.allclose(I_approx, I_true, atol=1e-6):
            print("✅ Correct inverse (A * A⁻¹ ≈ I)")
        else:
            print("❌ Incorrect inverse (A * A⁻¹ differs from I!)")

        print(f"‣ Error norm: {error_norm:.3e}")
        print(f"‣ Flops: {flops}")
        print(f"‣ Time: {elapsed*1000:.2f} ms")

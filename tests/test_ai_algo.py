import numpy as np


def test_calculate_matrix_C(ai_algorithm):
    tests_amount = 100
    for _ in range(tests_amount):
        A = np.random.randint(0, 10, size=(4, 5))
        B = np.random.randint(0, 10, size=(5, 5))

        # print("\nMacierz A (4x5):")
        # print(A)
        # print("\nMacierz B (5x5):")
        # print(B)

        expected_C = np.dot(A, B)

        try:
            actual_C = ai_algorithm(A, B)
        except Exception as e:
            print(f"\n❌ BŁĄD: Testowana funkcja zwróciła wyjątek: {e}")


        # np.array_equal jest bezpiecznym sposobem na porównanie dwóch tablic NumPy
        if np.array_equal(expected_C, actual_C):
            print("\n✅ TEST ZALICZONY: Wynik testowanej funkcji jest identyczny z np.dot.")
        else:
            print("\n❌ TEST NIEZALICZONY: Wyniki są różne.")
            difference = expected_C - actual_C
            print("\nRóżnica (Oczekiwany - Rzeczywisty):")
            print(difference)
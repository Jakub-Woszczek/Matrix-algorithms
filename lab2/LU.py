import numpy as np


def lu_recursive(A: np.ndarray, inversion_function, multiplication_function):
    """
    Block-recursive LU (no pivoting), A = L U, using user-provided:
      - multiplication_function(X, Y) -> (X@Y, flops_inc)
      - inversion_function(M)        -> (M^{-1}, flops_inc)

    Wymagania:
      - brak paddingu; przy n>2 wymagamy parzystego n na każdym poziomie splitu,
      - zakładamy nieosobliwość bloków wiodących (A11, U11, L11).

    Zwraca:
      L, U, flops
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.ndim == 2 and n == A.shape[1], "Matrix must be square"

    flops = 0  # globalny licznik w obrębie wywołania

    # --- pomocnicze opakowania na operacje z licznikami ---
    def mm(X, Y):
        nonlocal flops
        C, f = multiplication_function(X, Y)
        flops += int(f)
        return C

    def inv(M):
        nonlocal flops
        Minv, f = inversion_function(M, multiplication_function)
        flops += int(f)
        return Minv

    def msub(X, Y):
        """X - Y z doliczeniem FLOPs za odejmowanie element-po-elem."""
        nonlocal flops
        flops += X.size
        return X - Y

    # --- narzędzia blokowe ---
    def split(M):
        k = M.shape[0] // 2
        return M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]


    def join_blocks(L11, L21, L22, U11, U12, U22):
        """
        L = [[L11, 0_(k x (n-k))],
            [L21,       L22     ]]

        U = [[U11,   U12],
            [0_((n-k) x k), U22]]
        """
        k = L11.shape[0]
        n_minus_k = L22.shape[1]

        # L
        top_L = np.hstack((L11, np.zeros((k, n_minus_k))))
        bot_L = np.hstack((L21, L22))
        L = np.vstack((top_L, bot_L))

        # U
        top_U = np.hstack((U11, U12))
        bot_U = np.hstack((np.zeros((U22.shape[0], U11.shape[1])), U22))
        U = np.vstack((top_U, bot_U))
        return L, U

    # --- bazy ---
    def lu_1x1(M):
        L = np.array([[1.0]])
        U = np.array([[M[0, 0]]])
        return L, U

    def lu_2x2(M):
        nonlocal flops
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        u11 = a
        u12 = b
        l21 = c / u11
        flops += 1
        u22 = d - l21 * u12
        flops += 2
        L = np.array([[1.0, 0.0], [l21, 1.0]])
        U = np.array([[u11, u12], [0.0, u22]])
        return L, U

    # --- rekurencja ---
    def lu_rec(M):
        nonlocal flops
        nloc = M.shape[0]
        if nloc == 1:
            return lu_1x1(M)
        if nloc == 2:
            return lu_2x2(M)

        A11, A12, A21, A22 = split(M)

        # 1) A11 = L11 U11
        L11, U11 = lu_rec(A11)

        # 2) U12 = L11^{-1} A12   (tu korzystamy z przekazanej funkcji odwracania)
        L11_inv = inv(L11)
        U12 = mm(L11_inv, A12)

        # 3) L21 = A21 U11^{-1}
        U11_inv = inv(U11)
        L21 = mm(A21, U11_inv)

        # 4) S = A22 - L21 U12
        S = msub(A22, mm(L21, U12))

        # 5) S = L22 U22
        L22, U22 = lu_rec(S)

        # 6) Składanie
        return join_blocks(L11, L21, L22, U11, U12, U22)

    L, U = lu_rec(A)
    return L, U, flops

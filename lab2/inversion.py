import numpy as np

flops = 0


def recursion_inversion(A: np.ndarray, multiplication_function):
    """
    Returns matrix A inversion.
    :param A: square numpy array
    :param multiplication_function: function for matrix multiplication with signature
    (X, Y) -> (C, flops_inc)
    """

    global flops
    flops = 0

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.ndim == 2 and n == A.shape[1], "Matrix must be square"

    def split(M):
        k = M.shape[0] // 2
        return M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]

    def join(B11, B12, B21, B22):
        top = np.hstack((B11, B12))
        bottom = np.hstack((B21, B22))
        return np.vstack((top, bottom))

    def madd(X, Y):
        global flops
        flops += X.size
        return X + Y

    def msub(X, Y):
        global flops
        flops += X.size
        return X - Y

    def mm(X, Y):
        global flops
        C, f = multiplication_function(X, Y)
        flops += int(f)
        return C

    def inv1x1(M):
        global flops
        flops += 1
        return np.array([[1.0 / M[0, 0]]])

    def _inv(M):
        nloc = M.shape[0]
        if nloc == 1:
            return inv1x1(M)

        A11, A12, A21, A22 = split(M)

        # 1) A11^{-1}
        A11_inv = _inv(A11)

        # 2) S = A22 - A21 * A11^{-1} * A12
        P = mm(A21, A11_inv)
        Q = mm(P, A12)
        S22 = msub(A22, Q)

        # 3) S^{-1}
        S22_inv = _inv(S22)

        # 4)B11 = A11^{-1} + A11^{-1} * A12 * S^{-1} * A21 * A11^{-1}
        T1 = mm(A11_inv, A12)
        T2 = mm(T1, S22_inv)
        T3 = mm(T2, A21)
        T4 = mm(T3, A11_inv)
        B11 = madd(A11_inv, T4)

        # B12 = - A11^{-1} * A12 * S^{-1}
        U1 = mm(A11_inv, A12)
        U2 = mm(U1, S22_inv)
        B12 = -U2

        # B21 = - S^{-1} * A21 * A11^{-1}
        V1 = mm(S22_inv, A21)
        V2 = mm(V1, A11_inv)
        B21 = -V2

        # B22 = S^{-1}
        B22 = S22_inv

        return join(B11, B12, B21, B22)

    A_inv = _inv(A)
    return A_inv, flops

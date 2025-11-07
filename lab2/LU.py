import numpy as np

def determinant_via_lu(A: np.ndarray, inversion_function, multiplication_function):
    L, U, flops = lu_recursive(A, inversion_function, multiplication_function)
    detA = float(np.prod(np.diag(U)))
    return detA, L, U, flops

def lu_recursive(A: np.ndarray, inversion_function, multiplication_function):

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.ndim == 2 and n == A.shape[1], "Matrix must be square"

    flops = 0

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
        nonlocal flops
        flops += X.size
        return X - Y

    def split(M):
        k = M.shape[0] // 2
        return M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]


    def join_blocks(L11, L21, L22, U11, U12, U22):

        k = L11.shape[0]
        n_minus_k = L22.shape[1]

        top_L = np.hstack((L11, np.zeros((k, n_minus_k))))
        bot_L = np.hstack((L21, L22))
        L = np.vstack((top_L, bot_L))

        top_U = np.hstack((U11, U12))
        bot_U = np.hstack((np.zeros((U22.shape[0], U11.shape[1])), U22))
        U = np.vstack((top_U, bot_U))
        return L, U

    def lu_1x1(M):
        L = np.array([[1.0]])
        U = np.array([[M[0, 0]]])
        return L, U

    def lu_rec(M):
        nonlocal flops
        nloc = M.shape[0]
        if nloc == 1:
            return lu_1x1(M)

        A11, A12, A21, A22 = split(M)

        # 1) A11 = L11 U11
        L11, U11 = lu_rec(A11)

        # 2) U12 = L11^{-1} A12
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

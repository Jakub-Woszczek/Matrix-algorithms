import numpy as np


def recursive_gauss_elimination(
    A: np.ndarray,
    inversion_function,
    multiplication_function,
    lu_factor_function,
    b: np.ndarray | None = None,
):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.ndim == 2 and n == A.shape[1], "Matrix must be square"

    b_col = None
    if b is not None:
        b_arr = np.asarray(b, dtype=float)
        assert b_arr.ndim in (1, 2) and b_arr.shape[0] == n
        b_col = b_arr.reshape(n, 1)

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

    def lu(Ablk):
        nonlocal flops
        out = lu_factor_function(Ablk, inversion_function, multiplication_function)
        if len(out) == 3:
            L, U, f = out
            flops += int(f)
        else:
            L, U = out
        return L, U

    def msub(X, Y):
        nonlocal flops
        flops += X.size
        return X - Y

    def join_U(U11, U12, U22):
        k = U11.shape[0]
        top = np.hstack((U11, U12))
        bot = np.hstack((np.zeros((U22.shape[0], k)), U22))
        return np.vstack((top, bot))

    def base_1x1(M, b_local):
        U = M.copy()
        if b_local is None:
            return U, None
        return U, b_local / U[0, 0]

    def rec(M, b_loc):
        nloc = M.shape[0]
        if nloc == 1:
            return base_1x1(M, b_loc)

        k = nloc // 2
        A11, A12 = M[:k, :k], M[:k, k:]
        A21, A22 = M[k:, :k], M[k:, k:]
        b1 = b_loc[:k, :] if b_loc is not None else None
        b2 = b_loc[k:, :] if b_loc is not None else None

        # 1) [L11, U11] = LU(A11)
        L11, U11 = lu(A11)

        # 2) L11^{-1}, U11^{-1}
        L11_inv = inv(L11)
        U11_inv = inv(U11)

        # 3) C12 = L11^{-1} A12
        C12 = mm(L11_inv, A12)

        # 4) M21 = A21 U11^{-1}
        M21 = mm(A21, U11_inv)

        # 5) S = A22 - M21 C12
        S = msub(A22, mm(M21, C12))

        # 6) [Ls, Us] = LU(S)
        Ls, Us = lu(S)

        # 7) U = [[U11, C12],[0, Us]]
        U_big = join_U(U11, C12, Us)

        if b_loc is None:
            return U_big, None

        # 8) RHS1 = L11^{-1} b1
        RHS1 = mm(L11_inv, b1)

        # 9) v = M21 * RHS1
        v = mm(M21, RHS1)

        # 10) RHS2 = Ls^{-1} b2 - Ls^{-1} v
        Ls_inv = inv(Ls)
        RHS2 = msub(mm(Ls_inv, b2), mm(Ls_inv, v))

        # 11) x2 = Us^{-1} RHS2
        Us_inv = inv(Us)
        x2 = mm(Us_inv, RHS2)

        # 12) x1 = U11^{-1} (RHS1 - C12 x2)
        rhs_top = msub(RHS1, mm(C12, x2))
        x1 = mm(U11_inv, rhs_top)

        x = np.vstack((x1, x2))
        return U_big, x

    U, x = rec(A, b_col)
    return (U, flops) if b_col is None else (U, x.reshape(-1), flops)

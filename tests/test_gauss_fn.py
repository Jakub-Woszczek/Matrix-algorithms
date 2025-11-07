import numpy as np

def test_gauss_fn(
    gauss_fn,                 # callable(A, inversion_function, multiplication_function, lu_factor_function, b=None)
    inversion_function,
    multiplication_function,
    lu_factor_function,
    start=2, stop=30, step=1, seed=0, atol=1e-8
):
    """
    Test rekurencyjnej eliminacji Gaussa (blokowej, 'jak w PDF').

    Sprawdza:
      - kształty,
      - górnotrójkątność U,
      - zgodność z referencyjnym U_ref z klasycznej eliminacji bez pivotowania,
      - poprawność rozwiązania Ax=b (jeśli gauss_fn je zwraca).

    Parametry:
      gauss_fn: A, inv_fn, mm_fn, lu_fn, b=None -> (U, flops) lub (U, x, flops)
      inversion_function(M, multiplication_function) -> (M_inv, flops)
      multiplication_function(X, Y) -> (X@Y, flops)
      lu_factor_function(Ablk, multiplication_function, inversion_function) -> (L,U[,flops])
    """

    rng = np.random.default_rng(seed)
    name = getattr(gauss_fn, "__name__", str(gauss_fn))
    print(f"testing {name}")

    # prosta referencja: klasyczna eliminacja bez pivotowania, zwraca U
    def upper_no_pivot_reference(A):
        A = A.copy().astype(float)
        n = A.shape[0]
        for k in range(n - 1):
            pivot = A[k, k]
            if np.isclose(pivot, 0.0):
                raise np.linalg.LinAlgError("Zero pivot in reference GE (no pivoting).")
            for i in range(k + 1, n):
                m = A[i, k] / pivot
                A[i, k:] -= m * A[k, k:]
                A[i, k] = 0.0
        return A

    for n in range(start, stop, step):
        print(f"\n=== Test for {n}x{n} matrix ===")
        A = rng.normal(size=(n, n)).astype(float)
        b = rng.normal(size=(n,)).astype(float)

        # policz U_ref (może się wywalić dla pechowych macierzy bez pivotu)
        try:
            U_ref = upper_no_pivot_reference(A)
        except Exception as e:
            print(f"⏭️  skipped reference (no-pivot) failed: {e}")
            continue

        # uruchom testowaną funkcję (z i bez RHS)
        try:
            out_no_rhs = gauss_fn(A, inversion_function, multiplication_function, lu_factor_function, b=None)
            # (U, flops) albo (U, x, flops) gdy ktoś omyłkowo zwróci x
            if len(out_no_rhs) == 3:
                U_only, flops_no_rhs = out_no_rhs[0], out_no_rhs[2]
            else:
                U_only, flops_no_rhs = out_no_rhs[0], None
        except ValueError as e:
            print(f"⏭️  skipped recursive: {e}")
            continue
        except Exception as e:
            print(f"❌ exception (no RHS): {type(e).__name__}: {e}")
            continue

        # z RHS
        try:
            out_with_rhs = gauss_fn(A, inversion_function, multiplication_function, lu_factor_function, b=b)
            if len(out_with_rhs) == 3:
                U_rhs, x, flops_rhs = out_with_rhs
            else:
                # defensywnie: jeśli ktoś zwróci (U,x) bez flops
                U_rhs, x = out_with_rhs
                flops_rhs = None
        except Exception as e:
            print(f"❌ exception (with RHS): {type(e).__name__}: {e}")
            continue

        # weryfikacje
        ok_shape_no_rhs = U_only.shape == (n, n)
        ok_tri_no_rhs   = np.allclose(np.tril(U_only, -1), 0.0, atol=atol)
        close_ref_no_rhs = np.allclose(U_only, U_ref, atol=atol, rtol=0.0)
        rel_err_no_rhs = np.linalg.norm(U_only - U_ref, ord="fro") / (np.linalg.norm(U_ref, ord="fro") + 1e-15)

        ok_shape_rhs = U_rhs.shape == (n, n)
        ok_tri_rhs   = np.allclose(np.tril(U_rhs, -1), 0.0, atol=atol)
        close_ref_rhs = np.allclose(U_rhs, U_ref, atol=atol, rtol=0.0)
        rel_err_rhs = np.linalg.norm(U_rhs - U_ref, ord="fro") / (np.linalg.norm(U_ref, ord="fro") + 1e-15)

        # jeśli zwrócono x, sprawdź residual
        res_ok = False
        res_norm = np.nan
        if x is not None:
            x = x.reshape(-1)
            r = A @ x - b
            res_norm = np.linalg.norm(r, ord=2) / (np.linalg.norm(b, ord=2) + 1e-15)
            res_ok = res_norm <= 1e2 * atol  # lekko luźniej niż atol na elementach

        passed = (ok_shape_no_rhs and ok_tri_no_rhs and close_ref_no_rhs
                  and ok_shape_rhs and ok_tri_rhs and close_ref_rhs
                  and res_ok)

        if passed:
            f1 = f" | FLOPs(noRHS)={flops_no_rhs}" if flops_no_rhs is not None else ""
            f2 = f" | FLOPs(RHS)={flops_rhs}" if flops_rhs is not None else ""
            print(f"✅ OK (rel_err U={rel_err_no_rhs:.2e}/{rel_err_rhs:.2e}, resid={res_norm:.2e}){f1}{f2}")
        else:
            print(
                "❌ FAIL "
                f"(noRHS: shape={ok_shape_no_rhs}, tri={ok_tri_no_rhs}, close_ref={close_ref_no_rhs}, "
                f"rel_err={rel_err_no_rhs:.2e}; "
                f"RHS: shape={ok_shape_rhs}, tri={ok_tri_rhs}, close_ref={close_ref_rhs}, "
                f"rel_err={rel_err_rhs:.2e}, resid={res_norm:.2e})"
            )

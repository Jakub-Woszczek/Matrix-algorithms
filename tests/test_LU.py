import numpy as np


def test_lu_fn(
    lu_fn,
    inversion_function,
    multiplication_function,
    start=2,
    stop=30,
    step=1,
    seed=0,
    atol=1e-8,
):
    """
    Prosty test LU bez pivotowania:
      - dla n w [start, stop) co 'step' losuje A ~ N(0,1)
      - wywołuje lu_fn(A)
      - weryfikuje rekonstrukcję i własności trójkątne
    """

    rng = np.random.default_rng(seed)
    name = getattr(lu_fn, "__name__", str(lu_fn))
    print(f"testing {name}")

    for n in range(start, stop, step):
        print(f"\n=== Test for {n}x{n} matrix ===")
        A = rng.normal(size=(n, n)).astype(float)

        try:
            out = lu_fn(A, inversion_function, multiplication_function)
            L, U, flops = out

        except ValueError as e:
            # np. gdy implementacja wymaga parzystych rozmiarów (brak paddingu)
            msg = str(e).lower()
            print(f"❌ exception from test_LU: {type(e).__name__}: {e}")
            continue
        except Exception as e:
            print(f"❌ exception from test_LU: {type(e).__name__}: {e}")
            continue

        # Weryfikacje jak w poprzednich testach
        ok_shape = L.shape == (n, n) and U.shape == (n, n)
        ok_L_tri = np.allclose(np.triu(L, 1), 0.0, atol=atol)
        ok_U_tri = np.allclose(np.tril(U, -1), 0.0, atol=atol)
        ok_L_diag = np.allclose(np.diag(L), 1.0, atol=atol)

        A_hat = L @ U
        rel_err = np.linalg.norm(A - A_hat, ord="fro") / (
            np.linalg.norm(A, ord="fro") + 1e-15
        )

        if (
            ok_shape
            and ok_L_tri
            and ok_U_tri
            and ok_L_diag
            and np.allclose(A, A_hat, atol=atol, rtol=0)
        ):
            flops_str = f" | FLOPs={flops}" if flops is not None else ""
            print(f"✅ OK (rel_err={rel_err:.3e}){flops_str}")
        else:
            print(
                f"❌ FAIL "
                f"(shape={ok_shape}, triL={ok_L_tri}, triU={ok_U_tri}, diagL={ok_L_diag}, rel_err={rel_err:.3e})"
            )

import numpy as np

flops_strassen = 0


def strassen_multiplication(A, B):
    """
    Returns matrix multiplication of matrix A with matrix B.
    :param A: square numpy array
    :param B: square numpy array
    """
    global flops_strassen

    assert A.shape[1] == B.shape[0], "Incompatible shapes for multiplication"

    def split_matrix(matrix):
        """Splits np matrix into 4 pieces"""
        top, bottom = np.array_split(matrix, 2, axis=0)
        A11, A12 = np.array_split(top, 2, axis=1)
        A21, A22 = np.array_split(bottom, 2, axis=1)
        return A11, A12, A21, A22

    def naive_multiplication(A, B):
        global flops_strassen
        A_rows, A_cols = A.shape
        B_rows, B_cols = B.shape

        assert A_cols == B_rows, "Incompatible shapes for multiplication"

        result = np.empty((A_rows, B_cols))
        flops_strassen += A_rows * B_cols * (2 * B_rows - 1)

        for i in range(A_rows):
            for j in range(B_cols):
                s = 0
                for k in range(A_cols):
                    s += A[i, k] * B[k, j]
                result[i, j] = s

        return result

    def remove_row(A, B):
        global flops_strassen

        A_main = A[:-1, :-1]
        A_col = A[:-1, -1:]
        A_row = A[-1:, :-1]
        alpha = A[-1:, -1:]

        B_main = B[:-1, :-1]
        B_col = B[:-1, -1:]
        B_row = B[-1:, :-1]
        beta = B[-1:, -1:]

        top_left = req_multiplication(A_main, B_main) + naive_multiplication(
            A_col, B_row
        )
        flops_strassen += top_left.size
        top_right = naive_multiplication(A_main, B_col) + naive_multiplication(
            A_col, beta
        )
        flops_strassen += top_right.size
        bottom_left = naive_multiplication(A_row, B_main) + naive_multiplication(
            alpha, B_row
        )
        flops_strassen += bottom_left.size
        bottom_right = naive_multiplication(A_row, B_col) + naive_multiplication(
            alpha, beta
        )
        flops_strassen += bottom_right.size

        top = np.hstack((top_left, top_right))
        bottom = np.hstack((bottom_left, bottom_right))
        return np.vstack((top, bottom))

    def req_multiplication(A, B):
        """Multiplication of matrix A with matrix B using Strassen method, adding padding if needed"""
        global flops_strassen
        A_rows, A_cols = np.shape(A)
        B_rows, B_cols = np.shape(B)

        # Padding
        n = max(A_rows, A_cols, B_rows, B_cols)
        if n % 2 == 1:
            n += 1

        if (A_rows != n) or (A_cols != n) or (B_rows != n) or (B_cols != n):
            A_padded = np.zeros((n, n), dtype=A.dtype)
            B_padded = np.zeros((n, n), dtype=B.dtype)
            A_padded[:A_rows, :A_cols] = A
            B_padded[:B_rows, :B_cols] = B
            C_padded = req_multiplication(A_padded, B_padded)
            return C_padded[:A_rows, :B_cols]

        if A_rows == 2 and A_cols == 2 and B_rows == 2 and B_cols == 2:
            flops_strassen += 12
            return np.array(
                [
                    [
                        A[0][0] * B[0][0] + A[0][1] * B[1][0],
                        A[0][0] * B[0][1] + A[0][1] * B[1][1],
                    ],
                    [
                        A[1][0] * B[0][0] + A[1][1] * B[1][0],
                        A[1][0] * B[0][1] + A[1][1] * B[1][1],
                    ],
                ]
            )

        if A_rows % 2 == 1:
            return remove_row(A, B)

        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)

        P1 = req_multiplication(A11 + A22, B11 + B22)
        P2 = req_multiplication(A21 + A22, B11)
        P3 = req_multiplication(A11, B12 - B22)
        P4 = req_multiplication(A22, B21 - B11)
        P5 = req_multiplication(A11 + A12, B22)
        P6 = req_multiplication(A21 - A11, B11 + B12)
        P7 = req_multiplication(A12 - A22, B21 + B22)

        flops_strassen += 10 * A11.size

        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 - P2 + P3 + P6

        flops_strassen += 8 * P1.size

        upper = np.hstack((C11, C12))
        lower = np.hstack((C21, C22))
        return np.vstack((upper, lower))

    return req_multiplication(A, B), flops_strassen

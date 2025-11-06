import numpy as np


flops_binet = 0


def binet_multiplication(A, B):
    """
    Returns matrix multiplication of matrix A with matrix B.
    Works for both even and odd square matrices.
    :param A: square numpy array
    :param B: square numpy array
    """
    global flops_binet

    assert A.shape[1] == B.shape[0], "Incompatible shapes for multiplication"

    def split_matrix(matrix):
        """Splits np matrix into 4 blocks."""
        top, bottom = np.array_split(matrix, 2, axis=0)
        A11, A12 = np.array_split(top, 2, axis=1)
        A21, A22 = np.array_split(bottom, 2, axis=1)
        return A11, A12, A21, A22

    def naive_multiplication(A, B):
        global flops_binet

        A_rows, A_cols = A.shape
        B_rows, B_cols = B.shape

        assert A_cols == B_rows, "Incompatible shapes for multiplication"

        result = np.empty((A_rows, B_cols))
        flops_binet += A_rows * B_cols * (2 * B_rows - 1)

        for i in range(A_rows):
            for j in range(B_cols):
                s = 0

                for k in range(A_cols):
                    s += A[i, k] * B[k, j]
                result[i, j] = s

        return result

    def req_multiplication(A, B):
        global flops_binet
        A_rows, A_cols = np.shape(A)
        B_rows, B_cols = np.shape(B)

        if A_rows == 0 or A_cols == 0 or B_rows == 0 or B_cols == 0:
            return np.zeros((A_rows, B_cols), dtype=float)

        if A_rows == 1 or A_cols == 1 or B_rows == 1 or B_cols == 1:
            return naive_multiplication(A, B)

        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)

        C11 = req_multiplication(A11, B11) + req_multiplication(A12, B21)
        flops_binet += C11.size
        C12 = req_multiplication(A11, B12) + req_multiplication(A12, B22)
        flops_binet += C12.size
        C21 = req_multiplication(A21, B11) + req_multiplication(A22, B21)
        flops_binet += C21.size
        C22 = req_multiplication(A21, B12) + req_multiplication(A22, B22)
        flops_binet += C22.size

        top = np.hstack((C11, C12))
        bottom = np.hstack((C21, C22))
        return np.vstack((top, bottom))

    return req_multiplication(A, B), flops_binet

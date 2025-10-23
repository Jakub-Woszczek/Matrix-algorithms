import numpy as np


def binet_multiplication(A, B):
    """
    Returns matrix multiplication of matrix A with matrix B.
    :param A: square numpy array
    :param B: square numpy array
    """

    def split_matrix(matrix):
        """Splits np matrix into 4 pieces"""
        top, bottom = np.array_split(matrix, 2, axis=0)
        A11, A12 = np.array_split(top, 2, axis=1)
        A21, A22 = np.array_split(bottom, 2, axis=1)
        return A11, A12, A21, A22

    assert A.shape[1] == B.shape[0], "Incompatible shapes for multiplication"

    def req_multiplication(A, B):
        A_rows, A_cols = np.shape(A)
        B_rows, B_cols = np.shape(B)

        if A_rows == 1 or A_cols == 1 or B_rows == 1 or B_cols == 1:
            return np.dot(A, B)

        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)

        C11 = req_multiplication(A11, B11) + req_multiplication(A12, B21)
        C12 = req_multiplication(A11, B12) + req_multiplication(A12, B22)
        C21 = req_multiplication(A21, B11) + req_multiplication(A22, B21)
        C22 = req_multiplication(A21, B12) + req_multiplication(A22, B22)

        top = np.hstack((C11, C12))
        bottom = np.hstack((C21, C22))
        return np.vstack((top, bottom))

    return req_multiplication(A, B)

def binet_multiplication(A, B):
    """
    Returns matrix multiplication of matrix A with matrix B.
    """
    rows_a = len(A)  # number of rows in A
    columns_a = len(A[0])  # number of columns in A
    rows_b = len(B)  # number of rows in B
    columns_b = len(B[0])  # number of columns in B

    assert rows_a > 0
    assert rows_b > 0
    assert columns_a == rows_b, "Incompatible shapes for multiplication"

    assert all(len(row) == len(A[0]) for row in A), "A is not rectangular"
    assert all(len(row) == len(B[0]) for row in B), "B is not rectangular"

    output_matrix = [[0 for _ in range(columns_b)] for _ in range(rows_a)]

    for i, rowA in enumerate(A):

        for col in range(len(B[0])):
            for row in range(len(B)):
                output_matrix[i][col] += rowA[row] * B[row][col]

    return output_matrix

class MyMatrix:
    """
    Lightweight matrix view (window) into a larger NumPy array.
    """

    def __init__(self, matrix, min_row, max_row, min_col, max_col):
        self.matrix = matrix

        # Validate coordinates
        if not (0 <= min_row <= max_row <= matrix.shape[0]):
            raise ValueError("Row bounds outside matrix range")
        if not (0 <= min_col <= max_col <= matrix.shape[1]):
            raise ValueError("Column bounds outside matrix range")

        self.min_row = min_row
        self.max_row = max_row
        self.min_col = min_col
        self.max_col = max_col

    def get_view(self):
        """Return a NumPy *view* of the selected submatrix."""
        return self.matrix[self.min_row : self.max_row, self.min_col : self.max_col]

    def compress_matrix(self):
        """
        Split the matrix view into 4 quadrants (UL, UR, LL, LR).
        Returns a list of 4 MyMatrix objects.
        """
        mid_row = (self.min_row + self.max_row) // 2
        mid_col = (self.min_col + self.max_col) // 2

        return [
            # Upper-left
            MyMatrix(self.matrix, self.min_row, mid_row, self.min_col, mid_col),
            # Upper-right
            MyMatrix(self.matrix, self.min_row, mid_row, mid_col, self.max_col),
            # Lower-left
            MyMatrix(self.matrix, mid_row, self.max_row, self.min_col, mid_col),
            # Lower-right
            MyMatrix(self.matrix, mid_row, self.max_row, mid_col, self.max_col),
        ]

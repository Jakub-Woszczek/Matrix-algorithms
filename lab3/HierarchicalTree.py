import random

import numpy as np
from matplotlib import pyplot as plt

from lab3.image_magic.split_rgb import split_rgb


class HierarchicalTree:
    def __init__(self, image_path):
        self.image = plt.imread(image_path)
        self.root = None
        self.red_layer = None
        self.green_layer = None
        self.blue_layer = None

        self.red_layer, self.green_layer, self.blue_layer = split_rgb(self.image)

        print(self.red_layer)

    def create_tree(self, image_matrix):
        """
        :param image_matrix: numpy matrix
        :return:
        """
        max_row, max_col = image_matrix.shape
        my_matrix = MyMatrix(image_matrix, 0, max_row, 0, max_col)
        self.root = self.build_node(my_matrix)

    def build_node(self, my_matrix):

        node = Node()
        node.my_matrix = my_matrix
        view = my_matrix.get_view()
        s, v, d = np.linalg.svd(view)
        node.S, node.V, node.D = s, v, d  # Optional (im not sure if its necessary)

        # Base case
        if random.choice([True, False]):
            # TODO warunek rank i wartość osobliwa / macierz za mała do powdzielenia
            return node
        else:
            ul, ur, ll, lr = my_matrix.compress_matrix()
            ul_node = self.build_node(ul)
            ur_node = self.build_node(ur)
            ll_node = self.build_node(ll)
            lr_node = self.build_node(lr)
            node.set_children([ul_node, ur_node, ll_node, lr_node])

        return node


class Node:
    """
    Children:
      UpperLeft  - index 0
      UpperRight - index 1
      LowerLeft  - index 2
      LowerRight - index 3
    """

    def __init__(self):
        self.children = [None, None, None, None]
        self.my_matrix = None
        self.S, self.V, self.D = None, None, None

    def set_children(self, children):
        self.children = children

    def set_my_matrix(self, matrix):
        self.my_matrix = matrix


class MyMatrix:
    """
    Lightweight matrix view (window) into a larger NumPy array.

    Stores:
    1. Coordinates of a rectangular subregion:
       - min_row (inclusive)
       - max_row (exclusive)
       - min_col (inclusive)
       - max_col (exclusive)
       -> it mean that max_row=5 is not included, last row included is row=4 (trust me,I'm almost engineer)
       These define the slice: matrix[min_row:max_row, min_col:max_col]
    2. A reference (pointer) to the full matrix.
    3. Provides a .get() method returning a NumPy view into the matrix.
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

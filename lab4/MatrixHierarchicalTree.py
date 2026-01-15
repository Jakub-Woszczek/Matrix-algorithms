import numpy as np
from matplotlib import pyplot as plt

from lab3.MyMatrix import MyMatrix
from typing import Tuple


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
        self.U, self.S, self.VT = None, None, None

    def set_children(self, children):
        self.children = children

    def is_leaf(self):
        return all(ch is None for ch in self.children)


class MatrixHierarchicalTree:
    def __init__(self, delta, b_rank, min_size=4, delta_probe=256):
        self.b_rank = int(b_rank)
        self.min_size = int(min_size)
        self.delta = (
            delta  # może być float / int (indeks) / "median"/"mean"/"max"/"min"
        )
        self.delta_probe = int(delta_probe)  # rozmiar bloku do estymacji delta
        self.root = None

    def _compute_delta_from_probe(self, A: np.ndarray) -> float:
        """
        Liczymy SVD małego bloku (określone przez delta probe) bo całe svd cięzkie
        """
        n = A.shape[0]
        p = min(self.delta_probe, n)
        probe = A[:p, :p]

        U, S, VT = np.linalg.svd(probe, full_matrices=False)

        if isinstance(self.delta, (int, np.integer)):
            i = int(self.delta)
            i = max(0, min(i, S.size - 1))
            return float(S[i])

        if isinstance(self.delta, str):
            if self.delta == "median":
                return float(np.median(S))
            if self.delta == "mean":
                return float(np.mean(S))
            if self.delta == "max":
                return float(np.max(S))
            if self.delta == "min":
                return float(np.min(S))
            raise ValueError(f"Nieznany tryb delta: {self.delta}")

        return float(self.delta)

    def create_tree(self, matrix: np.ndarray) -> Node:
        max_row, max_col = matrix.shape
        my_matrix = MyMatrix(matrix, 0, max_row, 0, max_col)

        # ustal delta bez pełnego SVD całej macierzy
        self.delta = self._compute_delta_from_probe(matrix)
        self.root = self.build_node(my_matrix)
        print("Tree created with delta =", self.delta)
        return self.root

    def build_node(self, my_matrix: MyMatrix) -> Node:
        node = Node()
        node.my_matrix = my_matrix
        view = my_matrix.get_view()

        # pełne SVD bloku
        U, S, VT = np.linalg.svd(view, full_matrices=False)

        # odcinamy małe wartości osobliwe
        mask = S >= self.delta
        S_thr = S[mask]

        # ograniczamy rangę do b_rank
        k = min(self.b_rank, len(S_thr))
        S_used = S_thr[:k]
        U_used = U[:, :k]
        VT_used = VT[:k, :]

        node.U = U_used
        node.S = S_used
        node.VT = VT_used

        height, width = view.shape
        min_side = min(height, width)

        should_split = (
            min_side >= 2 * self.min_size and k == self.b_rank and len(S_thr) > 0
        )

        if not should_split:
            return node

        ul, ur, ll, lr = my_matrix.compress_matrix()
        node.set_children(
            [
                self.build_node(ul),
                self.build_node(ur),
                self.build_node(ll),
                self.build_node(lr),
            ]
        )
        return node

    def draw_partition(self, shape: Tuple[int, int]) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Drzewo nie zostało zbudowane.")
        h, w = shape
        canvas = np.full((h, w), 255, dtype=np.uint8)
        thickness = 1

        def draw_node(node: Node):
            m = node.my_matrix
            r0, r1, c0, c1 = m.min_row, m.max_row, m.min_col, m.max_col

            if node.is_leaf():
                canvas[r0 : r0 + thickness, c0:c1] = 0
                canvas[r1 - thickness : r1, c0:c1] = 0
                canvas[r0:r1, c0 : c0 + thickness] = 0
                canvas[r0:r1, c1 - thickness : c1] = 0
                return

            for ch in node.children:
                if ch is not None:
                    draw_node(ch)

        draw_node(self.root)
        return canvas

    def reconstruct_matrix(self) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Drzewo nie zostało zbudowane.")
        n, m = self.root.my_matrix.get_view().shape
        out = np.zeros((n, m), dtype=np.float64)
        self._reconstruct_node(self.root, out)
        return out

    def _reconstruct_node(self, node: Node, out_matrix: np.ndarray):
        if node.is_leaf():
            block = node.U @ (np.diag(node.S) @ node.VT)
            view = node.my_matrix
            out_matrix[view.min_row : view.max_row, view.min_col : view.max_col] = block
            return
        for ch in node.children:
            self._reconstruct_node(ch, out_matrix)


def h_mv_mult(root: Node, x: np.ndarray) -> np.ndarray:
    """
    y = A*x w oparciu o drzewo kompresji.
    """
    n = root.my_matrix.max_row - root.my_matrix.min_row
    y = np.zeros(n, dtype=np.float64)

    def rec(node: Node):
        my_matrix = node.my_matrix
        r0, r1, c0, c1 = (
            my_matrix.min_row,
            my_matrix.max_row,
            my_matrix.min_col,
            my_matrix.max_col,
        )

        if node.is_leaf():
            xs = x[c0:c1]  # fragment wektora
            # VT @ xs -> (k,)
            tmp = node.VT @ xs
            # diag(S) @ tmp  == S * tmp
            tmp2 = node.S * tmp
            # U @ tmp2 -> (rows,)
            y[r0:r1] += node.U @ tmp2
            return

        for ch in node.children:
            rec(ch)

    rec(root)
    return y


def _dense_block_from_node(node):
    """
    Zwraca gęstą macierz (np.ndarray) reprezentowaną przez node.
    """

    if node.is_leaf:
        if node.my_matrix is not None:
            return node.my_matrix.get_view()

        if node.U is not None and node.S is not None and node.VT is not None:
            return node.U @ (np.diag(node.S) @ node.VT)

        raise AttributeError("Leaf node nie ma my_matrix ani (U,S,VT).")

    ch = node.children
    if ch is None or len(ch) == 0:
        raise AttributeError("Internal node nie ma children.")

    if len(ch) != 4:
        raise ValueError(
            f"Nieobsługiwana liczba dzieci: {len(ch)} (spodziewałem się 4)."
        )

    A11 = _dense_block_from_node(ch[0])
    A12 = _dense_block_from_node(ch[1])
    A21 = _dense_block_from_node(ch[2])
    A22 = _dense_block_from_node(ch[3])

    return np.block([[A11, A12], [A21, A22]])


def h_add(A: Node, B: Node, compressor: MatrixHierarchicalTree) -> Node:
    """
    C = A + B
    Dla prostoty: składamy gęsto blok i kompresujemy procedurą z Zad. 3.
    """
    # blok A i B powinny mieć ten sam zakres
    dense = _dense_block_from_node(A) + _dense_block_from_node(B)

    local = MyMatrix(dense, 0, dense.shape[0], 0, dense.shape[1])
    return compressor.build_node(local)


def h_mult(A: Node, B: Node, compressor: MatrixHierarchicalTree) -> Node:
    """
    C = A * B
    """
    # jeśli liść/liść albo mieszany -> fallback dense
    if A.is_leaf() or B.is_leaf():
        dense = _dense_block_from_node(A) @ _dense_block_from_node(B)
        local = MyMatrix(dense, 0, dense.shape[0], 0, dense.shape[1])
        return compressor.build_node(local)

    # obie strony węzły
    A11, A12, A21, A22 = A.children
    B11, B12, B21, B22 = B.children

    # C11 = A11*B11 + A12*B21
    P1 = h_mult(A11, B11, compressor)
    P2 = h_mult(A12, B21, compressor)
    C11 = h_add(P1, P2, compressor)

    # C12 = A11*B12 + A12*B22
    P3 = h_mult(A11, B12, compressor)
    P4 = h_mult(A12, B22, compressor)
    C12 = h_add(P3, P4, compressor)

    # C21 = A21*B11 + A22*B21
    P5 = h_mult(A21, B11, compressor)
    P6 = h_mult(A22, B21, compressor)
    C21 = h_add(P5, P6, compressor)

    # C22 = A21*B12 + A22*B22
    P7 = h_mult(A21, B12, compressor)
    P8 = h_mult(A22, B22, compressor)
    C22 = h_add(P7, P8, compressor)

    # Sklejamy dzieci do nowego node:
    node = Node()
    node.set_children([C11, C12, C21, C22])
    C_dense = np.block(
        [
            [_dense_block_from_node(C11), _dense_block_from_node(C12)],
            [_dense_block_from_node(C21), _dense_block_from_node(C22)],
        ]
    )

    my_matrix = MyMatrix(C_dense, 0, C_dense.shape[0], 0, C_dense.shape[1])
    node = compressor.build_node(my_matrix)
    return node

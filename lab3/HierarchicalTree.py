import numpy as np
from matplotlib import pyplot as plt

from lab3.MyMatrix import MyMatrix
from lab3.image_magic.split_hsv import split_hsv


class HierarchicalTree:
    def __init__(self, image_path, delta, b_rank, min_size=4):
        self.image = plt.imread(image_path)

        # parametry sterujące podziałem

        self.b_rank = int(b_rank)  # maksymalna ranga bloku
        self.min_size = int(min_size)  # minimalny rozmiar bloku (wys / szer)
        self.delta = delta  # próg wartości osobliwych`
        self.root = None
        self.hue_layer = None
        self.saturation_layer = None
        self.value_layer = None

        self.hue_layer, self.saturation_layer, self.value_layer = split_hsv(self.image)

    def create_tree(self, image_matrix: np.ndarray):
        max_row, max_col = image_matrix.shape
        my_matrix = MyMatrix(image_matrix, 0, max_row, 0, max_col)

        view = my_matrix.get_view()
        U, S, VT = np.linalg.svd(view, full_matrices=False)
        print("S size:", S.size)

        if isinstance(self.delta, (int, float, np.integer, np.floating)):
            self.delta = S[int(self.delta)]
        if isinstance(self.delta, str):
            if self.delta == "median":
                self.delta = float(np.median(S))
            elif self.delta == "mean":
                self.delta = float(np.mean(S))
            elif self.delta == "max":
                self.delta = float(np.max(S))
            elif self.delta == "min":
                self.delta = float(np.min(S))

        self.root = self.build_node(my_matrix)
        print("Tree created with delta =", self.delta)
        return self.root

    def build_node(self, my_matrix):
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

        # warunki podziału
        should_split = (
            min_side >= 2 * self.min_size  # musi się dać sensownie podzielić
            and k == self.b_rank  # mamy "pełną" rangę więc jest jeszcze co kompresować
            and len(S_thr) > 0
            and S_used[-1] >= self.delta  # najmniejsza zachowana sigma nadal duża
        )

        if not should_split:
            # lisc
            return node

        # rekurencyjnie dzielimy na 4 podbloki
        ul, ur, ll, lr = my_matrix.compress_matrix()
        ul_node = self.build_node(ul)
        ur_node = self.build_node(ur)
        ll_node = self.build_node(ll)
        lr_node = self.build_node(lr)
        node.set_children([ul_node, ur_node, ll_node, lr_node])

        return node

    # rekonstrukcja obrazu z drzewa
    def reconstruct_channel(self, shape):

        # Rekonstruuje jeden kanal

        if self.root is None:
            raise RuntimeError("Drzewo nie zostało zbudowane. Wywołaj create_tree().")

        recon = np.zeros(shape, dtype=float)
        self._reconstruct_node(self.root, recon)
        return np.clip(recon, 0, 255).astype(np.uint8)

    def _reconstruct_node(self, node, out_matrix):
        # jeśli liść – wstawiamy skompresowany blok
        if node.is_leaf():
            block = node.U @ (np.diag(node.S) @ node.VT)
            view = node.my_matrix
            out_matrix[
                view.min_row : view.max_row,
                view.min_col : view.max_col,
            ] = block
            return

        # węzeł wewnętrzny – schodzimy rekurencyjnie
        for child in node.children:
            self._reconstruct_node(child, out_matrix)

    def draw_partition(self, shape):

        if self.root is None:
            raise RuntimeError("Drzewo nie zostało zbudowane.")

        h, w = shape
        canvas = np.full((h, w), 255, dtype=np.uint8)  # białe tło
        thickness = 1

        def draw_node(node: Node):
            m = node.my_matrix
            r0, r1, c0, c1 = m.min_row, m.max_row, m.min_col, m.max_col

            if node.is_leaf():
                # górna i dolna krawędź
                canvas[r0 : r0 + thickness, c0:c1] = 0
                canvas[r1 - thickness : r1, c0:c1] = 0
                # lewa i prawa krawędź
                canvas[r0:r1, c0 : c0 + thickness] = 0
                canvas[r0:r1, c1 - thickness : c1] = 0
                return

            # węzeł wewnętrzny, rysujemy jego dzieci
            for ch in node.children:
                if ch is not None:
                    draw_node(ch)

        draw_node(self.root)
        return canvas


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

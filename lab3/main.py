import os
import matplotlib

matplotlib.use("Agg")  # backend bez GUI
import matplotlib.pyplot as plt
from PIL import Image

from lab3.HierarchicalTree import HierarchicalTree
from lab3.image_magic.split_hsv import split_hsv

IMAGE_SAVE_PATH = r"lab3/processed_images/"
IMAGE_SAVE_PATH_REPORT = r"lab3/processed_images/report/"
IMAGE_READ_PATH = r"lab3/img.png"

image = Image.open(IMAGE_READ_PATH)
h, s, v = split_hsv(image)


"""
delta = liczba (tj ile wartości osobliwych ma być zachowanych w SVD)
lub
delta = "median" lub "mean" lub "max" lub "min"

b_rank = maksymalna ranga bloku
min_size = minimalny rozmiar bloku (wys / szer)

"""
# delta_set = ["median","mean","max","min"]
delta_set = ["mean", "max", "min"]
rank_set = [1, 4]

delta = "mean"
b_rank = 4
for b_rank in rank_set:
    for delta in delta_set:
        print(b_rank, delta)
        folder_name = f"rank_{b_rank}_delta_{delta}"
        full_path = f"{IMAGE_SAVE_PATH_REPORT}/{folder_name}/"
        os.makedirs(full_path, exist_ok=True)

        tree_h = HierarchicalTree(
            IMAGE_READ_PATH, delta=delta, b_rank=b_rank, min_size=4
        )
        tree_s = HierarchicalTree(
            IMAGE_READ_PATH, delta=delta, b_rank=b_rank, min_size=4
        )
        tree_v = HierarchicalTree(
            IMAGE_READ_PATH, delta=delta, b_rank=b_rank, min_size=4
        )

        tree_h.hue_layer = h
        tree_s.saturation_layer = s
        tree_v.value_layer = v

        tree_h.create_tree(tree_h.hue_layer)
        tree_s.create_tree(tree_s.saturation_layer)
        tree_v.create_tree(tree_v.value_layer)

        # rekonstrukcja
        recon_h = tree_h.reconstruct_channel(tree_h.hue_layer.shape)
        recon_s = tree_s.reconstruct_channel(tree_s.hue_layer.shape)
        recon_v = tree_v.reconstruct_channel(tree_v.hue_layer.shape)

        # wizualizacja każdego kanału
        Image.fromarray(recon_h).save(full_path + f"h_compressed.png")
        Image.fromarray(recon_s).save(full_path + f"s_compressed.png")
        Image.fromarray(recon_v).save(full_path + f"v_compressed.png")

        # składamy pełen obraz
        hsv_img = Image.merge(
            "HSV",
            (
                Image.fromarray(recon_h),
                Image.fromarray(recon_s),
                Image.fromarray(recon_v),
            ),
        )

        recon_img = hsv_img.convert("RGB")
        recon_img.save(full_path + "witcher_compressed.png")

        # wizualizacje podziału dla każdego kanału
        partition_img_h = tree_h.draw_partition(tree_h.hue_layer.shape)
        Image.fromarray(partition_img_h).save(
            full_path + "witcher_partition_structure_h.png"
        )

        partition_img_s = tree_s.draw_partition(tree_s.hue_layer.shape)
        Image.fromarray(partition_img_s).save(
            full_path + "witcher_partition_structure_s.png"
        )

        partition_img_v = tree_v.draw_partition(tree_v.hue_layer.shape)
        Image.fromarray(partition_img_v).save(
            full_path + "witcher_partition_structure_v.png"
        )

import matplotlib

matplotlib.use("Agg")# backend bez GUI
import matplotlib.pyplot as plt
from PIL import Image

from lab3.HierarchicalTree import HierarchicalTree
from lab3.image_magic.split_rgb import split_rgb

IMAGE_SAVE_PATH = r"lab3/processed_images/"
IMAGE_READ_PATH = r"lab3/image.png"

image = Image.open(IMAGE_READ_PATH)
r, g, b = split_rgb(image)




'''
delta = liczba (tj ile wartości osobliwych ma być zachowanych w SVD)
lub
delta = "median" lub "mean" lub "max" lub "min"

b_rank = maksymalna ranga bloku
min_size = minimalny rozmiar bloku (wys / szer)

'''
tree_r = HierarchicalTree(IMAGE_READ_PATH, delta="median" ,b_rank=4, min_size=4)
tree_g = HierarchicalTree(IMAGE_READ_PATH, delta="min", b_rank=4, min_size=4)
tree_b = HierarchicalTree(IMAGE_READ_PATH, delta=24, b_rank=4, min_size=4)

tree_r.red_layer = r
tree_g.green_layer = g
tree_b.blue_layer = b

tree_r.create_tree(tree_r.red_layer)
tree_g.create_tree(tree_g.green_layer)
tree_b.create_tree(tree_b.blue_layer)

# rekonstrukcja
recon_r = tree_r.reconstruct_channel(tree_r.red_layer.shape)
recon_g = tree_g.reconstruct_channel(tree_g.red_layer.shape)
recon_b = tree_b.reconstruct_channel(tree_b.red_layer.shape)

# składamy pełen obraz
recon_img = Image.merge(
    "RGB",
    (Image.fromarray(recon_r), Image.fromarray(recon_g), Image.fromarray(recon_b)),
)

# zapis porównania
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Skompresowany")
plt.imshow(recon_img)
plt.axis("off")
plt.tight_layout()
plt.savefig(IMAGE_SAVE_PATH + "cyberp_anime_comparison.png", dpi=150)
recon_img.save(IMAGE_SAVE_PATH + "cyberp_anime_compressed.png")

# wizualizacje podziału dla każdego kanału
partition_img_r = tree_r.draw_partition(tree_r.red_layer.shape)
plt.figure(figsize=(5, 5))
plt.title("Struktura podziału bloków – R")
plt.imshow(partition_img_r, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.tight_layout()
plt.savefig(IMAGE_SAVE_PATH + "cyberp_anime_partition_structure_red.png", dpi=150)

partition_img_g = tree_g.draw_partition(tree_g.red_layer.shape)
plt.figure(figsize=(5, 5))
plt.title("Struktura podziału bloków – G")
plt.imshow(partition_img_g, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.tight_layout()
plt.savefig(IMAGE_SAVE_PATH + "cyberp_anime_partition_structure_green.png", dpi=150)

partition_img_b = tree_b.draw_partition(tree_b.red_layer.shape)
plt.figure(figsize=(5, 5))
plt.title("Struktura podziału bloków – B")
plt.imshow(partition_img_b, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.tight_layout()
plt.savefig(IMAGE_SAVE_PATH + "cyberp_anime_partition_structure_blue.png", dpi=150)
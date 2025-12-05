import matplotlib

matplotlib.use("Agg")  # backend bez GUI
import matplotlib.pyplot as plt
from PIL import Image

from lab3.HierarchicalTree import HierarchicalTree
from lab3.image_magic.split_rgb import split_rgb

IMAGE_SAVE_PATH = r"lab3/processed_images/"
IMAGE_READ_PATH = r"lab3/cyberp_anime.png"

image = Image.open(IMAGE_READ_PATH)
r, g, b = split_rgb(image)


tree = HierarchicalTree(IMAGE_READ_PATH, delta=20, b_rank=4, min_size=4)

tree.create_tree(tree.red_layer)


recon_r = tree.reconstruct_channel(tree.red_layer.shape)
recon_img = Image.merge(
    "RGB",
    (Image.fromarray(recon_r), Image.fromarray(g), Image.fromarray(b)),
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Skompresowany (R kanał hierarchicznie)")
plt.imshow(recon_img)
plt.axis("off")
plt.tight_layout()

plt.savefig(IMAGE_SAVE_PATH + "cyberp_anime_comparison.png", dpi=150)
recon_img.save(IMAGE_SAVE_PATH + "cyberp_anime_compressed.png")

partition_img = tree.draw_partition(tree.red_layer.shape)
plt.figure(figsize=(5, 5))
plt.title("Struktura podziału bloków")
plt.imshow(partition_img, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.tight_layout()
plt.savefig(IMAGE_SAVE_PATH + "cyberp_anime_partition_structure.png", dpi=150)

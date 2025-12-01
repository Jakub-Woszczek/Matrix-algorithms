import matplotlib.pyplot as plt
from PIL import Image
from fontTools.misc.arrayTools import pointInRect

from lab3.image_magic.save_with_color import save_with_color
from lab3.image_magic.split_rgb import split_rgb

IMAGE_SAVE_PATH = r"lab3/processed_images/"
IMAGE_READ_PATH = r"lab3/cyberp_anime.png"

image = plt.imread(IMAGE_READ_PATH)
print(image.shape)
r, g, b = split_rgb(image)
print(r)

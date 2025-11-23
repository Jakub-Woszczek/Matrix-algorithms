from PIL import Image

from lab3.image_magic.save_with_color import save_with_color
from lab3.image_magic.split_rgb import split_rgb

IMAGE_SAVE_PATH = r"lab3/processed_images/"

image = Image.open(r"lab3/cyberp_anime.png")
r, g, b = split_rgb(image)
save_with_color(r, "anime", "blue", IMAGE_SAVE_PATH)

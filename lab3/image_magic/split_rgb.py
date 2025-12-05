import numpy as np
from PIL import Image

def split_rgb(image):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            img_255 = (255 * image).clip(0, 255).astype(np.uint8)
        else:
            img_255 = image
        image = Image.fromarray(img_255)

    r, g, b = image.convert("RGB").split()
    return np.array(r), np.array(g), np.array(b)
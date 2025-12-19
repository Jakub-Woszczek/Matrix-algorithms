import numpy as np
from PIL import Image


def split_hsv(image):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            img_255 = (255 * image).clip(0, 255).astype(np.uint8)
        else:
            img_255 = image
        image = Image.fromarray(img_255)

    h, s, v = image.convert("HSV").split()
    return np.array(h), np.array(s), np.array(v)

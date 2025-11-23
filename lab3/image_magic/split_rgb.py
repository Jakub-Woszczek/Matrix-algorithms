def split_rgb(image):
    r, g, b = image.convert("RGB").split()

    return r, g, b

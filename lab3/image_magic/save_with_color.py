from PIL import Image


def save_with_color(color_channel, name, color: str, save_path):
    """
    Creates and saves an RGB image highlighting a single color channel.

    Parameters:
    - color_channel (PIL.Image.Image): A single-channel (grayscale) image representing the intensity of the color.
    - name (str): Base name for the saved image file.
    - color (str): Color channel to highlight ('red', 'green', or 'blue').
    - save_path (str): Directory path where the image will be saved.

    The resulting image will have the specified color channel in full intensity while the other channels are set to zero.
    """

    zero = Image.new("L", color_channel.size, 0)

    if color.lower() == "red":
        colored_image = Image.merge("RGB", (color_channel, zero, zero))
    elif color.lower() == "green":
        colored_image = Image.merge("RGB", (zero, color_channel, zero))
    elif color.lower() == "blue":
        colored_image = Image.merge("RGB", (zero, zero, color_channel))
    else:
        raise ValueError("Color must be 'red', 'green', or 'blue'")

    colored_image.save(save_path + name + f"_{color}" + ".png", "PNG")

import numpy as np


def color_quantization_img(
    img: np.ndarray,
    mandatory_colors: list[tuple[float, float, float]],
) -> np.ndarray:
    """
    Performs color quantization on the `img` to set all colours equal to the closest
    colour in  `mandatory_colors`. `mandatory_colors` is a list of colors in the form
    of a tuple (red, green, blue) with colour values indicated by integer values
    inbetween 0 and 255 or float values between 0 and 1.

    :return: a quantized version of the input image with only color values in
    `mandatory_colors`
    """
    img = img[:, :, 0:3]  # keep only RGB channels
    w, h, _ = img.shape
    img = np.reshape(img, (w * h, 3))

    img = color_quantization(img, mandatory_colors)

    img = np.reshape(img, (w, h, 3))
    return img


def color_quantization(
    arr: np.ndarray,
    mandatory_colors: list[tuple[float, float, float]],
) -> np.ndarray:
    """
    Performs color quantization on the `arr` to set all colours equal to the closest
    colour in  `mandatory_colors`. `mandatory_colors` is a list of colors in the form
    of a tuple (red, green, blue) with colour values indicated by integer values
    inbetween 0 and 255 or float values between 0 and 1.

    :return: a quantized version of the input image with only color values in
    `mandatory_colors`
    """
    pixel_dict = {}
    for i, pixel in enumerate(arr):
        pixel_tuple = tuple(pixel)
        if pixel_tuple in pixel_dict.keys():
            pixel_dict[tuple(pixel_tuple)].append(i)
        else:
            pixel_dict[pixel_tuple] = [i]

    # Find the closest class color to each pixels
    # and assign them to this class
    for pixel_value, indices in pixel_dict.items():
        distances = np.sum((mandatory_colors - np.array(pixel_value)) ** 2, axis=1)
        arr[indices] = mandatory_colors[np.where(distances == np.min(distances))[0][0]]

    return arr

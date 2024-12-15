import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def load_image(filepath):
    """
    Load an image from disk into a numpy array.

    Args:
        filepath (str): Path to the image file.

    Returns:
        np.ndarray: Image as a numpy array with dimensions (height, width, channels).
    """
    return np.array(Image.open(filepath))


def save_image(image, filepath):
    """
    Save a numpy array as an image to disk.

    Args:
        image (np.ndarray): Image data as a numpy array with dimensions (height, width, channels).
        filepath (str): Path to save the image file.
    """
    Image.fromarray(image).save(filepath)


def display_image(image, title=None, extend_range=False):
    """
    Display an image for visual inspection.

    Args:
        image (np.ndarray): Image data as a numpy array with dimensions (height, width, channels).
        title (str, optional): Title to display above the image.
        extend_range (bool): If True, check and normalize image values if they're outside [0, 255].
                           Useful for displaying images with values outside standard range.

    Notes:
        Normalization is only applied if extend_range=True AND image values are outside [0, 255].
    """
    display_img = image.copy()

    if extend_range:
        # Check if values are outside standard image range
        if np.min(image) < 0 or np.max(image) > 255:
            # Normalize image to [0, 255] range
            display_img = image - np.min(image)
            if np.max(display_img) > 0:
                display_img = display_img * 255 / np.max(display_img)
    if display_img.dtype == np.float64 and np.max(display_img) > 1:
        display_img = display_img.astype(np.uint8)
    plt.imshow(display_img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def gaussian_kernel(size=5):
    """
    Create a 2D Gaussian kernel using Pascal's triangle coefficients.

    Args:
        size (int): Size of the kernel. Must be odd and >= 1.

    Returns:
        np.ndarray: 2D Gaussian kernel with dimensions (size, size).

    Raises:
        ValueError: If size is even or less than 1.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("Kernel size must be odd and >= 1")

    # Generate Pascal's triangle row
    def pascal_row(n):
        row = [1]
        for k in range(n):
            row.append(row[k] * (n - k) // (k + 1))
        return np.array(row)

    # Get the middle row of Pascal's triangle
    n = (size - 1) // 2
    kernel_1d = pascal_row(2 * n)

    # Normalize and create 2D kernel
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    return kernel_2d


def reduce(image, kernel_size=5):
    """
    Reduce the image size by applying Gaussian blur and subsampling.

    Args:
        image (np.ndarray): Input image as a numpy array with dimensions (height, width, channels).
        kernel_size (int): Size of the Gaussian kernel. Must be odd and >= 1.

    Returns:
        np.ndarray: Reduced image with dimensions approximately (height/2, width/2, channels).
    """
    kernel = gaussian_kernel(kernel_size)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred[::2, ::2]


def expand(image, shape, kernel_size=5):
    """
    Expand the image size by upsampling and applying Gaussian blur.

    Args:
        image (np.ndarray): Input image as a numpy array with dimensions (height, width, channels).
        shape (tuple): Desired shape of the expanded image (height, width, channels).
        kernel_size (int): Size of the Gaussian kernel. Must be odd and >= 1.

    Returns:
        np.ndarray: Expanded image with dimensions (height, width, channels).
    """
    expanded = np.zeros(shape)
    expanded[::2, ::2] = image
    kernel = gaussian_kernel(kernel_size) * 4
    return cv2.filter2D(expanded, -1, kernel)


def generate_gaussian_pyramid(image, levels, same_size=True, kernel_size=5):
    """
    Generate a Gaussian pyramid for an image.

    Args:
        image (np.ndarray): Input image as a numpy array with dimensions (height, width, channels).
        levels (int): Number of levels in the pyramid.
        same_size (bool): If True, all images in the pyramid will have the same size as the original image.
        kernel_size (int): Size of the Gaussian kernel. Must be odd and >= 1.

    Returns:
        list: List of numpy arrays representing the Gaussian pyramid, each with dimensions (height, width, channels).
    """
    pyramid = [image]
    for _ in range(1, levels):
        image = reduce(pyramid[-1], kernel_size)
        if same_size:
            image = expand(image, pyramid[0].shape, kernel_size)
        pyramid.append(image)
    return pyramid


def generate_laplacian_pyramid(image, levels, same_size=True, kernel_size=5):
    """
    Build a Laplacian pyramid for an image.

    Args:
        image (np.ndarray): Input image as a numpy array with dimensions (height, width, channels).
        levels (int): Number of levels in the pyramid.
        same_size (bool): If True, all pyramid levels will have the same size as input image.
        kernel_size (int): Size of the Gaussian kernel. Must be odd and >= 1.

    Returns:
        list: List of numpy arrays representing the Laplacian pyramid, each with dimensions:
             - If same_size=True: All levels have dimensions (height, width, channels)
             - If same_size=False: Each level is half the size of the previous level
    """
    # Get Gaussian pyramid
    gaussian_pyr = generate_gaussian_pyramid(image, levels, same_size, kernel_size)
    laplacian_pyr = []

    # Build Laplacian pyramid
    for i in range(levels - 1):
        if same_size:
            laplacian_pyr.append((gaussian_pyr[i] - gaussian_pyr[i + 1]))
        else:
            expanded = expand(gaussian_pyr[i + 1], gaussian_pyr[i].shape, kernel_size)
            laplacian_pyr.append(gaussian_pyr[i] - expanded)

    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr

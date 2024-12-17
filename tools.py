import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def image_as_rgb(image):
    """
    Convert any image to RGB format in range [0, 255] as uint8.
    
    Args:
        image (np.ndarray): Input image with dimensions:
            - (height, width) for grayscale
            - (height, width, 1) for single channel
            - (height, width, 3) for RGB
            - (height, width, 4) for RGBA
            Values can be in any range and any numeric dtype
    
    Returns:
        np.ndarray: RGB image with dimensions (height, width, 3) in range [0, 255] as uint8
        
    Raises:
        ValueError: If image has unexpected number of channels
    """
    # Handle different channel configurations
    if len(image.shape) == 2:  # Grayscale
        rgb_image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        rgb_image = image[:, :, :3]
    elif image.shape[2] == 1:  # Single channel
        rgb_image = np.concatenate([image] * 3, axis=2)
    elif image.shape[2] == 3:  # Already RGB
        rgb_image = image
    else:
        raise ValueError(f"Unexpected number of channels: {image.shape[2]}")

    # Handle different value ranges
    if rgb_image.dtype != np.uint8:
        if np.min(rgb_image) < 0 or np.max(rgb_image) > 255:
            rgb_image = rgb_image - np.min(rgb_image)
            if np.max(rgb_image) > 0:
                rgb_image = rgb_image * 255 / np.max(rgb_image)
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
    
    return rgb_image


def load_image(filepath):
    """
    Load an image from disk into a numpy array in RGB format.

    Args:
        filepath (str): Path to the image file.

    Returns:
        np.ndarray: Image as a numpy array with dimensions (height, width, 3) in RGB format.
    """
    img = np.array(Image.open(filepath).convert('RGB'))
    return image_as_rgb(img)

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
        extend_range (bool): Deprecated parameter, kept for backward compatibility.
    """
    display_img = image_as_rgb(image)
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


def reconstruct_image(pyramid, kernel_size=5):
    """
    Reconstruct an image from its Laplacian pyramid.
    
    Args:
        pyramid (list): List of numpy arrays representing the Laplacian pyramid.
            Each array has dimensions (height, width, channels) where:
            - height and width may vary by level if pyramid was not created with same_size=True
            - channels is typically 3 for RGB images
        kernel_size (int): Size of the Gaussian kernel used for expansion. Must be odd and >= 1.
    
    Returns:
        np.ndarray: Reconstructed image with dimensions (height, width, channels) matching
                   the dimensions of the first level of the pyramid.
    
    Raises:
        ValueError: If pyramid is empty or kernel_size is invalid
    """
    if not pyramid:
        raise ValueError("Pyramid cannot be empty")
    
    # Start with the smallest level (Gaussian residual)
    reconstructed = pyramid[-1].copy()
    
    # Work up the pyramid, expanding and adding each level
    for level in reversed(pyramid[:-1]):
        # Expand the current reconstruction to match the size of the next level
        reconstructed = expand(reconstructed, level.shape, kernel_size)
        # Add the Laplacian detail from this level
        reconstructed = reconstructed + level
        
    return reconstructed

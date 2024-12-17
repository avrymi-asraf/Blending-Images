import numpy as np
from tools import (
    generate_laplacian_pyramid,
    generate_gaussian_pyramid,
    reconstruct_image,
    image_as_rgb,
)


def create_square_mask(height, width, center_x, center_y, size):
    """
    Creates a square mask for image blending.

    Args:
        height (int): Height of the mask
        width (int): Width of the mask
        center_x (int): X-coordinate of square center
        center_y (int): Y-coordinate of square center
        size (int): Size of the square side

    Returns:
        numpy.ndarray: Binary mask of shape (height, width) with values in {0, 1}
    """
    mask = np.zeros((height, width))
    half_size = size // 2
    x_start = max(0, center_x - half_size)
    x_end = min(width, center_x + half_size)
    y_start = max(0, center_y - half_size)
    y_end = min(height, center_y + half_size)
    mask[y_start:y_end, x_start:x_end] = 1
    return mask


def create_half_mask(height, width, vertical=True):
    """
    Creates a half image mask for blending.

    Args:
        height (int): Height of the mask
        width (int): Width of the mask
        vertical (bool): If True, splits vertically, else horizontally

    Returns:
        numpy.ndarray: Binary mask of shape (height, width) with values in {0, 1}
    """
    mask = np.zeros((height, width))
    if vertical:
        mask[:, : width // 2] = 1
    else:
        mask[: height // 2, :] = 1
    return mask


def blend_pyramids(lap_pyr1, lap_pyr2, mask_pyr):
    """
    Blends corresponding levels of two Laplacian pyramids using a Gaussian mask pyramid.

    Args:
        lap_pyr1 (list): First Laplacian pyramid, each element shape (h, w, 3)
        lap_pyr2 (list): Second Laplacian pyramid, each element shape (h, w, 3)
        mask_pyr (list): Gaussian pyramid of the mask, each element shape (h, w)

    Returns:
        list: Blended pyramid with same structure as input pyramids
    """
    blended_pyr = []
    for lap1, lap2, mask in zip(lap_pyr1, lap_pyr2, mask_pyr):
        # Expand mask to match image channels
        mask_3ch = np.expand_dims(mask, axis=2)
        mask_3ch = np.repeat(mask_3ch, 3, axis=2)
        # Blend the current level
        blended = lap1 * mask_3ch + lap2 * (1 - mask_3ch)
        blended_pyr.append(blended)
    return blended_pyr


def blend_images(img1, img2, mask, num_levels=6, kernel_size=5):
    """
    Blend two images using a mask and Laplacian pyramid.

    Args:
        img1 (numpy.ndarray): First image of shape (h, w, 3) with values in [0, 1]
        img2 (numpy.ndarray): Second image of shape (h, w, 3) with values in [0, 1]
        mask (numpy.ndarray): Binary mask of shape (h, w) with values in {0, 1}
        num_levels (int): Number of pyramid levels
        kernel_size (int): Size of the Gaussian kernel. Must be odd and >= 1

    Returns:
        numpy.ndarray: Blended image of shape (h, w, 3) with values in [0, 1]

    Raises:
        ValueError: If images have different shapes or invalid dimensions
    """
    # Validate inputs
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    if len(img1.shape) != 3 or img1.shape[2] != 3:
        raise ValueError("Images must be RGB with shape (h, w, 3)")

    # Generate Laplacian pyramids for both images
    lap_pyr1 = generate_laplacian_pyramid(img1, num_levels, False, kernel_size)
    lap_pyr2 = generate_laplacian_pyramid(img2, num_levels, False, kernel_size)

    # Generate Gaussian pyramid for the mask
    mask_pyr = generate_gaussian_pyramid(mask, num_levels, False, kernel_size)

    # Blend pyramids
    blended_pyr = blend_pyramids(lap_pyr1, lap_pyr2, mask_pyr)

    # Reconstruct the final image
    blended_img = reconstruct_image(blended_pyr, kernel_size)

    # Clip values to ensure they're in [0, 1]
    return image_as_rgb(blended_img)

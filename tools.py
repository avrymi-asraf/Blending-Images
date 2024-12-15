import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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

def display_image(image):
    """
    Display an image for visual inspection.
    
    Args:
        image (np.ndarray): Image data as a numpy array with dimensions (height, width, channels).
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def gaussian_blur(image, sigma=1):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array with dimensions (height, width, channels).
        sigma (float): Standard deviation for Gaussian kernel.
        
    Returns:
        np.ndarray: Blurred image with dimensions (height, width, channels).
    """
    return gaussian_filter(image, sigma=(sigma, sigma, 0))

def generate_gaussian_pyramid(image, levels, same_size=True):
    """
    Generate a Gaussian pyramid for an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array with dimensions (height, width, channels).
        levels (int): Number of levels in the pyramid.
        same_size (bool): If True, all images in the pyramid will have the same size as the original image.
        
    Returns:
        list: List of numpy arrays representing the Gaussian pyramid, each with dimensions (height, width, channels).
    """
    pyramid = [image]
    for _ in range(1, levels):
        image = gaussian_blur(image, sigma=2)
        image = image[::2, ::2]
        if same_size:
            padded_image = np.zeros_like(pyramid[0])
            padded_image[::2, ::2] = image
            image = gaussian_blur(padded_image, sigma=2)*4
        pyramid.append(image)
    return pyramid

def generate_laplacian_pyramid(image, levels, same_size=False):
    """
    Generate a Laplacian pyramid for an image.
    
    Args:
        image (np.ndarray): Input image as a numpy array with dimensions (height, width, channels).
        levels (int): Number of levels in the pyramid.
        same_size (bool): If True, all images in the pyramid will have the same size as the original image.
        
    Returns:
        list: List of numpy arrays representing the Laplacian pyramid, each with dimensions (height, width, channels).
    """
    gaussian_pyramid = generate_gaussian_pyramid(image, levels, same_size)
    laplacian_pyramid = []
    for i in range(levels - 1):
        next_level = np.repeat(np.repeat(gaussian_pyramid[i + 1], 2, axis=0), 2, axis=1)
        if same_size:
            padded_next_level = np.zeros_like(gaussian_pyramid[0])
            padded_next_level[::2, ::2] = next_level
            next_level = gaussian_blur(padded_next_level, sigma=2)
        laplacian = gaussian_pyramid[i] - next_level[:gaussian_pyramid[i].shape[0], :gaussian_pyramid[i].shape[1]]
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

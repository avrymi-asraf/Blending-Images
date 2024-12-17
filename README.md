# Image Blending Exercise

Implementation of image blending techniques for seamless transitions and hybrid image creation.

## Required Functions

### Tools
* [x] Load image to numpy array, `load_image`
* [x] Save image to disk, `save_image`
* [x] Display images for visual inspection, `display_image`
* [x] Generate Gaussian kernel, `gaussian_kernel`
* [x] Reduce image size, `reduce`
* [x] Expand image size, `expand`
* [x] Generate Gaussian Pyramid, `generate_gaussian_pyramid`
* [x] Generate Laplacian Pyramid, `generate_laplacian_pyramid`
* [x] Reconstruct image from Laplacian pyramid, `reconstruct_image`
* [x] Convert image to RGB format, `image_as_rgb`

### Task 1: Seamless Image Blending
The blending works by:

Breaking down each image into frequency bands (Laplacian pyramid)
Breaking down the mask into different scales (Gaussian pyramid)
Blending corresponding frequencies using the appropriate mask scale
Reconstructing the final image by combining all blended frequencies

This approach handles transitions between images at different frequency levels, creating smoother, more natural-looking blends than simple alpha blending.
#### Functions
* [ ] Blends corresponding levels of two Laplacian pyramids, Gaussian mask pyramid `blend_pyramids`
* [ ] Blend two images using a mask and Laplacian pyramid, `blend_images`
* [ ] Create squere mask for blending, `create_square_mask`
* [ ] Create half image mask for blending, `create_half_mask`

### Task 2: Hybrid Image Creation
* [ ] Create a hybrid image combining low frequencies from one image and high frequencies from another, `create_hybrid_image`
* [ ] Apply frequency-based filtering (low-pass and high-pass), `apply_frequency_filter`
* [ ] Adjust and visualize frequency domain representations, `visualize_frequency_components`
* [ ] Validate hybrid image results for clarity and contrast, `validate_hybrid_image`

## Input/Output Specifications

### Image Format
- Input images should be in RGB format (3 channels)
- Pixel values are normalized to range [0, 1]
- Images are represented as numpy arrays with shape (height, width, channels)

### Pyramid Specifications
- Gaussian pyramid: List of increasingly blurred images
- Laplacian pyramid: List of band-pass filtered images
- Each level reduces resolution by factor of 2

# Image Blending Tools

## Core Functions

### Image Handling Functions

- `image_as_rgb(image)`: Converts any image format to RGB in range [0, 255] as uint8
  - Handles grayscale, RGB, RGBA, and single-channel images
  - Automatically normalizes values to [0, 255] range
  - Converts to uint8 dtype

- `ensure_rgb(image)`: Ensures image is in RGB format (uses image_as_rgb internally)

- `load_image(filepath)`: Loads an image file into a numpy array in RGB format

- `save_image(image, filepath)`: Saves a numpy array as an image file

- `display_image(image, title=None)`: Displays an image with optional title
  - Automatically handles normalization via image_as_rgb


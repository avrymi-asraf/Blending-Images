# %% [markdown]
# # Image Blending Interface
# This notebook demonstrates the use of various image processing functions from `tools.py`.

# %%
import plotly.express as px
import numpy as np
from tools import (
    load_image,
    save_image,
    display_image,
    generate_gaussian_pyramid,
    generate_laplacian_pyramid,
    expand,
    reduce,
    gaussian_kernel,
)
from blending import create_square_mask, create_half_mask, blend_images
import treescope

tr = lambda *x: treescope.display(*x)

# %%
treescope.basic_interactive_setup()

# %%
# Load an image
image = load_image("data//hawk_tuah.png")
# display_image(image)

# Generate Laplacian pyramid
# %%
laplacian_pyramid = generate_laplacian_pyramid(image, levels=10, kernel_size=61)
for level, img in enumerate(laplacian_pyramid):
    print(f"Laplacian Pyramid Level {level}, dimensions: {img.shape}")
    display_image(img, extend_range=True)


# %%
levels = 10
image = load_image("data//hawk_tuah.webp")
gaussian_pyramid = generate_gaussian_pyramid(
    image, levels=levels, kernel_size=21, same_size=False
)
for i in range(levels - 1):
    display_image(
        gaussian_pyramid[i], title=f"Gaussian Pyramid Level {i}", extend_range=True
    )

# %%
image = load_image("data//hawk_tuah.webp")
laplacian_pyramid = generate_laplacian_pyramid(image, levels=30, kernel_size=71)
out = laplacian_pyramid[0]
for i in range(1, len(laplacian_pyramid)):
    out = out + laplacian_pyramid[i]
    display_image(out, title=f"Reconstructed Image at Level {i}", extend_range=True)
display_image(out, title="Reconstructed Image")

# %% [markdown]
# ## Image Blending Examples

# %%
# Load two images for blending
image1 = load_image("data//red-apple.png")
image2 = load_image("data//green-apple.png")  # Assuming you have a second image

# Ensure images are the same size
height, width = image1.shape[:2]

# %% [markdown]
# ### 1. Square Mask Blending Example

# %%
# Create square mask
square_mask = create_square_mask(
    height, width, center_x=width // 2, center_y=height // 2, size=200
)
display_image(square_mask, title="Square Mask")

# Blend images with square mask
square_blended = blend_images(image1, image2, square_mask, num_levels=16,kernel_size=21)
display_image(square_blended, title="Square Mask Blending Result")

# %% [markdown]
# ### 2. Vertical Half Mask Blending Example

# %%
# Create vertical half mask
vertical_mask = create_half_mask(height, width, vertical=True)
display_image(vertical_mask, title="Vertical Half Mask")

# Blend images with vertical mask
vertical_blended = blend_images(image1, image2, vertical_mask, num_levels=6)
display_image(vertical_blended, title="Vertical Half Blending Result")

# %% [markdown]
# ### 3. Horizontal Half Mask Blending Example

# %%
# Create horizontal half mask
horizontal_mask = create_half_mask(height, width, vertical=False)
display_image(horizontal_mask, title="Horizontal Half Mask")

# Blend images with horizontal mask
horizontal_blended = blend_images(image1, image2, horizontal_mask, num_levels=6)
display_image(horizontal_blended, title="Horizontal Half Blending Result")

# %% [markdown]
# ### 4. Comparison of Original Images and Results

# %%
# Display original images side by side
display_image(image1, title="Original Image 1")
display_image(image2, title="Original Image 2")

# Display all blending results
display_image(square_blended, title="Square Mask Blend")
display_image(vertical_blended, title="Vertical Half Blend")
display_image(horizontal_blended, title="Horizontal Half Blend")

# %%
from tools import load_image, display_image
from blending import (
    generate_laplacian_pyramid,
    generate_gaussian_pyramid,
    blend_pyramids,
    reconstruct_image,
    create_square_mask,
    create_half_mask,
)

# %%
img1 = load_image("data//red-apple.png")
img2 = load_image("data//green-apple.png")
num_levels = 6
mask = create_half_mask(img1.shape[0], img1.shape[1], vertical=True)
# %%
lap_pyr1 = generate_laplacian_pyramid(img1, num_levels, False)
lap_pyr2 = generate_laplacian_pyramid(img2, num_levels, False)

# Generate Gaussian pyramid for the mask
mask_pyr = generate_gaussian_pyramid(mask, num_levels, False)

# Blend pyramids
blended_pyr = blend_pyramids(lap_pyr1, lap_pyr2, mask_pyr)

# Reconstruct the final image
blended_img = reconstruct_image(blended_pyr)

# %%

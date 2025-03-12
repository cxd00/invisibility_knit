# pip install Pillow
import PIL
from PIL import Image

import numpy as np
import random
from collections import defaultdict

# parameters:
# filename --- str, name of the image
# num_colors --- int, number of colors expected
# num_samples --- int, number of samples to take per color

# returns:
# num_colors x num_samples x 2 (x,y) coordinates for control points
def sample_n_per_color(filename, num_samples):
    img = PIL.Image.open(filename)
    # img.show()

    width, height = img.size
    # print(width)
    # print(height)

    # Convert the image to 'RGB' mode if it's not (JPG does not support transparency)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    pix = img.load()

    # Get palette of colors
    # Quantize down to num_color palettised image using *"Fast Octree"* method
    # TODO: this quantization shouldn't need to be repeated
    # TODO: explore other quantization methods
    num_colors = len(img.getcolors())
    # print(num_colors)
    q = img.quantize(colors=num_colors,method=2)
    palette = np.array(q.getpalette())
    palette = palette.reshape(num_colors, 3)
    # print(palette)
    
    # Start a list of pixels of each color
    pixels_by_color = []
    # print(pixels_by_color)
    for idx, color in enumerate(palette):
        pixels_by_color.append([])
        for x in range(width):
            for y in range(height):
                pixel = np.array(pix[x, y])
                if np.array_equal(pixel, color):
                    pixels_by_color[idx].append([x, y])
            # print(pixel)
        pixels_by_color[idx] = np.array(pixels_by_color[idx])
    # pixels_by_color = np.array(pixels_by_color)
    # print(pixels_by_color.shape)

    # Sample pixels by color
    # For each color:
    samples_by_color = []
    for idx, candidates in enumerate(pixels_by_color):
        # Pick 60 pixels at random (with replacement)
        # random_samples = candidates[np.random.randint(candidates.shape[0], size=num_samples), :]
        # print(candidates)
        random_samples = [random.choice(candidates) for _ in range(num_samples)]
        # print(len(random_samples))
        samples_by_color.append(np.array(random_samples))
    samples_by_color = np.array(samples_by_color)
    # Add [0, 1) (random) to each pixel (from top left corner)
    randoms = np.random.rand(num_colors, num_samples, 2)
    # print(samples_by_color[0][:3])
    samples_by_color = samples_by_color + randoms

    # Scale from 0 to 1
    # for idx, color in enumerate(palette):

    # print(samples_by_color[0][:3])
    return samples_by_color, palette


if __name__ == "__main__":
    # palette = [[194, 192, 172], [157, 178, 194], [106, 115, 84], [83, 100, 116], [41, 62, 85], [46, 114, 75]]
    samples_by_color = sample_n_per_color("road.png", 60)
    # pixels_by_color = np.array(pixels_by_color)

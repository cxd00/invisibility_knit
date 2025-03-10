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
def sample_n_per_color(filename, num_colors, num_samples):
    img = PIL.Image.open(filename)
    # img.show()

    width, height = img.size
    ratio = 4
    # print(width)
    # print(height)

    # Convert the image to 'RGB' mode if it's not (JPG does not support transparency)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    pix = np.array(img)

    # Get palette of colors
    # Quantize down to num_color palettised image using *"Fast Octree"* method
    # TODO: this quantization shouldn't need to be repeated
    # TODO: explore other quantization methods
    q = img.quantize(colors=num_colors,method=2)
    palette = np.array(q.getpalette())
    palette = palette.reshape(num_colors, 3)
    
    # Start a list of pixels of each color
    pixels_by_color = defaultdict(list)
    # pixels_by_color = np.zeros(palette.shape[0])
    # print(pixels_by_color)
    for idx, color in enumerate(palette):
        # pixels_by_color.append([])
        for y in range(height):
            for x in range(width):
                pixel = np.array(pix[y, x])
                if np.array_equal(pixel, color):
                    pixels_by_color[idx].append(np.array([(y+np.random.rand())/height, (x+np.random.rand())/width]))
                    # pixels_by_color[idx].append(np.array([x, y]))
        # print("color", idx, len(pixels_by_color[idx]))

    # Sample pixels by color
    # For each color:
    samples_by_color = []
    for idx in pixels_by_color:
        candidates = np.array(pixels_by_color[idx])
        if len(candidates) < num_samples:
            raise Exception(f"not enough pixels for sampling in color {idx}")
        # Pick 60 pixels at random (with replacement)
        random_samples = candidates[np.random.randint(candidates.shape[0], size=num_samples), :]
        samples_by_color.append(np.clip(np.array(random_samples), 0,1))
    samples_by_color = np.array(samples_by_color)
    # Add [0, 1) (random) to each pixel (from top left corner)
    # randoms = np.random.rand(num_colors, num_samples, 2)/0.001
    # print(samples_by_color[0][:3])
    # samples_by_color = samples_by_color + randoms
    # samples_by_color[:,:,0] = samples_by_color[:,:,0]/height
    # samples_by_color[:,:,1] = samples_by_color[:,:,1]/width
    # print(samples_by_color[0][:3])
    return samples_by_color, palette


if __name__ == "__main__":
    # palette = [[194, 192, 172], [157, 178, 194], [106, 115, 84], [83, 100, 116], [41, 62, 85], [46, 114, 75]]
    samples_by_color = sample_n_per_color("road.png", 6, 60)
    # pixels_by_color = np.array(pixels_by_color)

# pip install Pillow
import PIL
from PIL import Image
import subprocess

import numpy as np
import random
from collections import defaultdict

palette = [
    (28, 25, 29),    # 91% Black
    (217, 123, 129), # 86% Puce
    (178, 8, 3),     # 90% Transport Red
    (157, 186, 186), # 89% Silver
    (31, 34, 59),    # 89% Prussian Blue
    (226, 213, 161), # 94% Vanilla
    (220, 224, 227), # 94% Anti-Flash White
    (245, 237, 233), # 97% Seashell
    (85, 71, 167),   # 84% Amethyst
    (60, 106, 151),  # 91% Steel Blue
    (125, 157, 184), # 92% Air Force Blue
    (252, 94, 0),    # 93% International Orange
    (0, 155, 138),   # 99% Bondi Blue
    (90, 84, 81),    # 90% Russet
    (191, 45, 86),   # 90% Cerise
    (116, 64, 180),  # 84% Amethyst
    (222, 181, 84),   # 88% Old Gold (#DEB454)
    (232, 148, 0),    # 95% Gamboge (#E89400)
    (198, 41, 0),      # 94% Titian (#C62900)
    (46, 153, 113),    # Sea green (#2E9971)
]

# PIL palette format is: [R, G, B, R, G, B, ...] and it's 256 colors long
# Our 20 color palette needs to be multiplied 12 times + 16 more
# TODO -- don't hardcode this my friend
PIL_palette = palette * 12 + palette[:16*3]

# parameters:
# filename --- str, name of the image
# new_width/height -- int, desired new proportions of image. Crop or expand if needed.
# num_colors --- int, number of colors expected
# TODO -- change to user chosen range of colors, from LESS COLORFUL to VERY COLORFUL?

# returns nothing, saves a new image to be sampled
def pixelate(orig_filename, new_width, new_height, num_colors):
    old_img = PIL.Image.open(orig_filename)

    # Convert the image to 'RGB' mode if it's not (JPG does not support transparency)
    if old_img.mode != 'RGB':
        old_img = img.convert('RGB')

    # old_img.load() # necessary?

    # 1. Resize image
    # TODO -- crop as needed. currently doesn't maintain aspect ratio
    # https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    resized_img = old_img.resize([new_width, new_height])

    resized_img.save("resized_" + orig_filename)

    # 2. Reduce number of colors AND map to our palette WITHOUT dithering

    # Establish palette for (A) and (B) approaches:
    # Reference: https://stackoverflow.com/questions/29433243/convert-image-to-specific-palette-using-pil-without-dithering
    # palette_img = Image.new('P', (16, 16))
    # palette_img.putpalette(PIL_palette)
    # palette_img.load() # necessary?
    # palette_img.convert('RGB')

    # (A) Naive approach: use PIL's built in quantization function
    # Problem: does dithering
    # n_color_img = new_img.convert('RGB').quantize(colors=6, method=0, kmeans=0, palette=palette_img)
    # methods: 0 - median cut, 1 - maximum coverage, 2 - fast octree

    # (B) Next attempt: try PIL convert to avoid dithering
    # new_img = resized_img.convert(mode="P", dither=0, palette=palette_img)
    # the 0 above means turn OFF dithering making solid colors
    # For the “P” mode, this method translates pixels through the palette.

    # Problem: doesn't seem to actually apply our palette
    #...or limit colors, even in adaptive palette, e.g.:
    # new_img = resized_img.convert(mode="P", dither=0, palette="Palette.ADAPTIVE", colors=2)
    # colors – Number of colors to use for the ADAPTIVE palette. 
    # Works differently when you call on instance .im
    # new_img = resized_img.im.convert("P", 0, palette_img.im)

    # new_img.show()

    # (C) Alternative to PIL: Imagemagick

    # Write "map.png" that is a 24x1 pixel image with one pixel for each colour
    # TODO -- fix so it generates 20 x 1, not too big. Ended up fixing by hand.
    # entries = len(palette)
    # resnp   = np.arange(entries,dtype=np.uint8).reshape(len(palette),1)
    # resim = Image.fromarray(resnp, mode='P')
    # resim.putpalette(palette)
    # resim.save('map.png')

    # Use Imagemagick to remap to palette saved above in 'map.png'
    # magick lion.png +dither -quantize Lab -remap map.png result.png
    subprocess.run(['magick', "resized_" + orig_filename, '+dither', '-quantize', 'Lab', '-remap', 'map.png', '-colors', str(num_colors), 'remapped_' + orig_filename])
    return

# TODO: consider using PIL histogram to make a smarter color selection
# TODO: alternatively, use pixelit or Stack Overflow's example of a pixel by pixel approach, e.g. MCMC, to pick closest colors
# TODO: add function to adjust blobbiness in pixelation

# parameters:
# filename --- str, name of the image
# num_colors --- int, number of colors expected
# num_samples --- int, number of samples to take per color

# returns:
# num_colors x num_samples x 2 (x,y) coordinates for control points
# TODO: also return names/RGB codes of colors
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
    name_str = "orig_test.jpg"
    pixelate(name_str, 216, 85, 6)
    samples_by_color = sample_n_per_color("remapped_" + name_str, 6, 60)
    # pixels_by_color = np.array(pixels_by_color)
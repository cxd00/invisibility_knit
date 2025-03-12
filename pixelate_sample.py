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

if __name__ == "__main__":
    name_str = "control.png"
    pixelate(name_str, 216, 85, 6)
    samples_by_color = sample_n_per_color("remapped_" + name_str, 6, 60)
    # pixels_by_color = np.array(pixels_by_color)
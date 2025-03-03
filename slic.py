# use slic to generate parameters for an input image so that it could be optimized by the generator

import cv2
import numpy as np
import skimage.segmentation as segmentation
from skimage.color import label2rgb
import matplotlib.pyplot as plt

def slic_segmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    segments = segmentation.slic(image, n_segments=100, compactness=10, sigma=1)    
    return segments, image

def visualize_segments(image, segments, save_path=None):
    # Create a visualization of the segments
    segmented_image = label2rgb(segments, image, kind='avg')
    
    # Display the original and segmented images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(segmented_image)
    ax2.set_title('Segmented Image')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def generate_parameters(image_path, save_path=None):
    segments, image = slic_segmentation(image_path)
    visualize_segments(image, segments, save_path)
    return segments

if __name__ == "__main__":
    image_path = "test_image.png"
    segments = generate_parameters(image_path, "segmented_output.png")
    print(f"Number of segments: {len(np.unique(segments))}")

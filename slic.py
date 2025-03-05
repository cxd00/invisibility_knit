# use slic to generate parameters for an input image so that it could be optimized by the generator

import cv2
import numpy as np
import skimage.segmentation as segmentation
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def slic_segmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    segments = segmentation.slic(image, n_segments=100, compactness=10, sigma=1)    
    return segments, image

def visualize_segments(image, segments, save_path=None):
    # Create a visualization of the segments
    segmented_image = label2rgb(segments, image, kind='avg')
    # Add segment boundaries
    segments_with_boundaries = segmentation.mark_boundaries(image, segments, color=(1, 1, 1))
    
    # Display the original and segmented images side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(segmented_image)
    ax2.set_title('Segmented Image')
    ax2.axis('off')
    
    ax3.imshow(segments_with_boundaries)
    ax3.set_title('Segments with Boundaries')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def generate_parameters(image_path, save_path=None):
    segments, image = slic_segmentation(image_path)
    visualize_segments(image, segments, save_path)
    return segments

def compute_distances(features, centroids):
    """Compute distances between pixels and cluster centroids."""
    n_pixels = features.shape[0]
    n_clusters = centroids.shape[0]
    
    # Reshape for broadcasting
    features_expanded = features.unsqueeze(1)  # [N, 1, D]
    centroids_expanded = centroids.unsqueeze(0)  # [1, K, D]
    
    # Compute Euclidean distances
    distances = torch.sum((features_expanded - centroids_expanded) ** 2, dim=2)  # [N, K]
    return distances

def differentiable_slic(image, n_segments=200, compactness=10.0, temperature=0.1):
    """
    Differentiable SLIC implementation
    Args:
        image: torch.Tensor of shape [H, W, C]
        n_segments: number of superpixels
        compactness: trade-off between color and spatial proximity
        temperature: temperature for softmax (lower = harder assignments)
    Returns:
        parameters: cluster centroids
        assignments: soft assignment matrix
    """
    if not torch.is_tensor(image):
        image = torch.from_numpy(image).float()
    
    H, W, C = image.shape
    n_pixels = H * W
    
    # Initialize grid of evenly spaced centroids
    grid_h = int(np.sqrt(n_segments))
    grid_w = int(np.sqrt(n_segments))
    h_step = H // grid_h
    w_step = W // grid_w
    
    # Create features matrix: [y, x, r, g, b]
    y_coords = torch.arange(H).float().unsqueeze(1).repeat(1, W)
    x_coords = torch.arange(W).float().unsqueeze(0).repeat(H, 1)
    
    # Scale spatial coordinates
    y_coords = y_coords / H * compactness
    x_coords = x_coords / W * compactness
    
    # Combine spatial and color features
    features = torch.cat([
        y_coords.unsqueeze(-1),
        x_coords.unsqueeze(-1),
        image
    ], dim=-1)
    
    # Reshape features to [N, D]
    features = features.reshape(n_pixels, -1)
    
    # Initialize centroids
    centroids = []
    for h in range(grid_h):
        for w in range(grid_w):
            y = h * h_step + h_step // 2
            x = w * w_step + w_step // 2
            centroid = features[y * W + x]
            centroids.append(centroid)
    centroids = torch.stack(centroids)
    
    # Iterative refinement
    for _ in range(10):  # Number of iterations
        # Compute distances
        distances = compute_distances(features, centroids)
        
        # Soft assignments using softmax
        assignments = F.softmax(-distances / temperature, dim=1)  # [N, K]
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        weights_sum = assignments.sum(dim=0).unsqueeze(1)  # [K, 1]
        for k in range(centroids.shape[0]):
            new_centroids[k] = (features * assignments[:, k:k+1]).sum(dim=0) / weights_sum[k]
        
        centroids = new_centroids
    
    return centroids, assignments

def gumbel_softmax(logits, temperature=0.3, hard=False, eps=1e-10):
    """
    Sample from Gumbel-Softmax distribution
    Args:
        logits: [N, K] tensor of log probabilities
        temperature: softmax temperature
        hard: if True, take argmax, but differentiate w.r.t. soft sample
        eps: epsilon for numerical stability
    Returns:
        [N, K] sample from Gumbel-Softmax distribution
    """
    # Sample from Gumbel distribution
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    
    # Softmax
    y_soft = gumbels.softmax(dim=-1)
    
    if hard:
        # Straight through trick
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    
    return ret

def reconstruct_image(centroids, assignments, original_shape, temperature=0.5, n_colors=6):
    """
    Reconstruct image from centroids and soft assignments with discrete color sampling
    Args:
        centroids: cluster centroids [K, D]
        assignments: soft assignment matrix [N, K]
        original_shape: tuple of (H, W, C)
        temperature: temperature for Gumbel-Softmax
        n_colors: number of colors per channel
    Returns:
        reconstructed image
    """
    H, W, C = original_shape
    
    # Extract color features from centroids (last 3 dimensions)
    color_centroids = centroids[:, -3:]  # [K, 3]
    
    # Define the color palette from HEX values
    color_palette = torch.tensor([
        [0.835, 0.486, 0.525],  # #d57c86
        [0.325, 0.392, 0.455],  # #536474
        [0.122, 0.118, 0.110],  # #1f1e1c
        [0.843, 0.792, 0.765],  # #d7cac3
        [0.616, 0.698, 0.761],  # #9db2c2
        [0.349, 0.208, 0.455],  # #593574
        [0.863, 0.090, 0.031],  # #dc1708
        [0.925, 0.855, 0.753],  # #ecdac0
        [0.929, 0.761, 0.576],  # #edc293
        [0.761, 0.753, 0.675],  # #c2c0ac
        [0.906, 0.408, 0.184],  # #e7682f
        [0.157, 0.118, 0.196],  # #281e32
        [0.416, 0.451, 0.329],  # #6a7354
        [0.059, 0.259, 0.224],  # #0f4239
        [0.161, 0.243, 0.333],  # #293e55
        [0.349, 0.227, 0.435],  # #593a6f
        [0.827, 0.859, 0.812],  # #d3dbcf
        [0.180, 0.447, 0.294],  # #2e724b
        [0.047, 0.612, 0.761],  # #0c9cc2
        [0.749, 0.839, 0.796],  # #bfd6cb
    ])
    
    # Compute squared Euclidean distances and scale them to make logits more peaked
    scaling_factor = 50.0  # Adjust this to control how strongly we prefer closer colors
    color_distances = torch.cdist(color_centroids, color_palette, p=2)  # [K, num_colors]
    color_logits = -scaling_factor * color_distances ** 2  # Convert distances to logits
    
    # Sample discrete colors using Gumbel-Softmax
    color_samples = gumbel_softmax(color_logits, temperature=temperature, hard=True)  # [K, num_colors]
    
    # Convert samples to RGB values
    discrete_colors = torch.mm(color_samples, color_palette)  # [K, 3]
    
    # Weighted sum based on assignments
    reconstructed = torch.mm(assignments, discrete_colors)  # [N, 3]
    
    # Reshape back to image
    reconstructed = reconstructed.reshape(H, W, C)
    
    return reconstructed

def generate_differentiable_parameters(image_path, save_path=None, n_colors=6):
    """Generate differentiable SLIC parameters and visualize results"""
    # Read and preprocess image
    image = cv2.imread(image_path)
    print("Image shape:", image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image)
    
    # Generate parameters
    centroids, assignments = differentiable_slic(image_tensor)
    print("Centroids shape:", centroids.shape)
    print("Centroids:", centroids)
    print("Assignments shape:", assignments.shape)
    print("Assignments:", assignments)
    
    # Get hard assignments for visualization
    hard_assignments = assignments.argmax(dim=1).reshape(image.shape[0], image.shape[1])
    segments_with_boundaries = segmentation.mark_boundaries(
        image, 
        hard_assignments.detach().numpy(), 
        color=(1, 1, 1)
    )
    
    # Reconstruct image with discrete colors
    reconstructed = reconstruct_image(centroids, assignments, image.shape, n_colors=n_colors)
    
    if save_path:
        # Visualize original, segmented with boundaries, and reconstructed
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(segments_with_boundaries)
        ax2.set_title('Segments with Boundaries')
        ax2.axis('off')
        
        ax3.imshow(reconstructed.detach().numpy())
        ax3.set_title(f'Reconstructed Image\n({n_colors} colors per channel)')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    return centroids, assignments

if __name__ == "__main__":
    image_path = "lighthouse_img.png"
    centroids, assignments = generate_differentiable_parameters(
        image_path, 
        "differentiable_segmented_output.png"
    )
    print(f"Number of clusters: {centroids.shape[0]}")

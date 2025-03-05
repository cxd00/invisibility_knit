# use slic to generate parameters for an input image so that it could be optimized by the generator

# Standard library imports
import os
import sys
import time
import math
from datetime import datetime
import gc

# Third-party imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
from tqdm import tqdm
from easydict import EasyDict

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.nn import parameter
from torch.autograd import Variable, Function
import torchvision
from torchvision import transforms

# PyTorch3D imports
import pytorch3d as p3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    cameras,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    AmbientLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    BlendParams,
    TexturesUV
)

# scikit-image imports
import skimage.segmentation as segmentation
from skimage.color import label2rgb

# Local imports
from generator import *
from load_data import *
from tps import *
from arch.yolov3_models import YOLOv3Darknet
from yolo2.darknet import Darknet
from color_util import *
from train_util import *
import pytorch3d_modify as p3dmd
import mesh_utils
import utils

# Add path for demo utils functions 
sys.path.append(os.path.abspath(''))

class SLICGenerator(torch.nn.Module):
    def __init__(self, image, n_segments=200, compactness=10.0, temperature=0.1):
        super().__init__()
        self.n_segments = n_segments
        self.compactness = compactness
        self.temperature = temperature
        
        # Initialize centroids using the input image
        with torch.no_grad():
            init_centroids, init_assignments = self.differentiable_slic(
                image, 
                self.n_segments, 
                self.compactness, 
                self.temperature,
                requires_grad=True
            )
        
        # Create learnable parameters
        self.centroids = torch.nn.Parameter(init_centroids.clone())
        self.assignments = torch.nn.Parameter(init_assignments.clone())
        self.color_weights = torch.nn.Parameter(torch.ones(6) / 6)  # For 6 colors
    
    def forward(self, image):
        # Compute new centroids and assignments using current parameters
        new_centroids, new_assignments = self.differentiable_slic(
                image, 
                self.n_segments, 
                self.compactness, 
                self.temperature,
                requires_grad=True)
        
        # Store the computed values (but don't assign to parameters)
        self._current_centroids = new_centroids
        self._current_assignments = new_assignments
        
        return new_centroids, new_assignments
    
    def reconstruct(self, image_shape):
        """
        Reconstruct image using current centroids and computed assignments
        Args:
            image_shape: tuple of (H, W, C)
        Returns:
            reconstructed image
        """
        # Use the most recently computed values
        reconstructed = self._reconstruct_image(
            self._current_centroids,
            self._current_assignments,
            image_shape,
            temperature=self.temperature
        )
        return reconstructed

    def _compute_distances(self, features, centroids):
        """Compute distances between pixels and cluster centroids."""
        
        # Reshape for broadcasting
        features_expanded = features.unsqueeze(1)  # [N, 1, D]
        centroids_expanded = centroids.unsqueeze(0)  # [1, K, D]
        
        # Compute Euclidean distances
        distances = torch.sum((features_expanded - centroids_expanded) ** 2, dim=2)  # [N, K]
        return distances

    def _create_features(self, image, n_segments, compactness):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).float()
        
        # Move image to the same device as the model
        device = image.device
        
        # Convert from [B, C, H, W] to [H, W, C]
        if len(image.shape) == 4:  # If we have a batch dimension
            image = image[0]  # Take first image from batch
            image = image.permute(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
            
        print("image shape after permute: ", image.shape)
        H, W, C = image.shape
        
        # Create features matrix: [y, x, r, g, b]
        y_coords = torch.arange(H, device=device).float().unsqueeze(1).repeat(1, W)
        x_coords = torch.arange(W, device=device).float().unsqueeze(0).repeat(H, 1)
        
        # Scale spatial coordinates
        y_coords = y_coords / H * compactness
        x_coords = x_coords / W * compactness
        
        # Create feature matrix combining spatial and color information
        features = torch.cat([
            y_coords.reshape(-1, 1),
            x_coords.reshape(-1, 1),
            image.reshape(-1, C)
        ], dim=1)  # [H*W, 5] tensor with [y, x, r, g, b]

        return features

    def differentiable_slic(self, image, n_segments=200, compactness=10.0, temperature=0.1, requires_grad=True):
        """
        Differentiable SLIC implementation with optional gradient tracking
        """
        # First, create image features matrix
        features = self._create_features(image, n_segments, compactness)
        device = features.device  # Get the device from features
        
        # Convert from [B, C, H, W] to [H, W, C]
        if len(image.shape) == 4:  # If we have a batch dimension
            image = image[0]  # Take first image from batch
            image = image.permute(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
            
        H, W, C = image.shape

        # Initialize grid of evenly spaced centroids
        grid_h = int(np.sqrt(n_segments))
        grid_w = int(np.sqrt(n_segments))
        h_step = H // grid_h
        w_step = W // grid_w

        # Initialize centroids with gradients if required
        centroids = []
        for h in range(grid_h):
            for w in range(grid_w):
                y = h * h_step + h_step // 2
                x = w * w_step + w_step // 2
                centroid = torch.tensor([
                    y / H * compactness,
                    x / W * compactness,
                    image[y, x, 0],
                    image[y, x, 1],
                    image[y, x, 2]
                ], requires_grad=requires_grad, device=device)  # Add device here
                centroids.append(centroid)
        centroids = torch.stack(centroids)
        
        # Iterative refinement
        for _ in range(10):  # Number of iterations
            # Compute distances using the features matrix
            distances = self._compute_distances(features, centroids)
            
            # Soft assignments using softmax
            assignments = self.gumbel_softmax(distances, temperature=temperature, hard=False)  # [N, K]
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            weights_sum = assignments.sum(dim=0).unsqueeze(1)  # [K, 1]
            for k in range(centroids.shape[0]):
                new_centroids[k] = (features * assignments[:, k:k+1]).sum(dim=0) / weights_sum[k]
            
            centroids = new_centroids
        
        return centroids, assignments

    def gumbel_softmax(self,logits, temperature=1.0, hard=False):
        """
        Sample from Gumbel-Softmax distribution
        Args:
            logits: [N, K] tensor of log probabilities
            temperature: softmax temperature
            hard: if True, take argmax, but differentiate w.r.t. soft sample
        Returns:
            [N, K] sample from Gumbel-Softmax distribution
        """
        # Sample from Gumbel distribution
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
        
        # Softmax
        y_soft = gumbels.softmax(dim=-1)
        
        if hard:
            # just take argmax
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        
        return ret

    def _reconstruct_image(self, centroids, assignments, original_shape, temperature=0.5, n_colors=6):
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
        if len(original_shape) == 4:  # If we have a batch dimension
            # original_shape is [B, C, H, W]
            B, C, H, W = original_shape
        else:
            H, W, C = original_shape.shape
            
        device = centroids.device  # Get the device from centroids
        
        # Extract color features from centroids (last 3 dimensions)
        color_centroids = centroids[:, -3:]  # [K, 3]
        
        # Define the color palette from HEX values of yarn
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
        ], device=device)
        
        # Compute squared Euclidean distances and scale them to make logits more peaked
        scaling_factor = 50.0  # Adjust this to control how strongly we prefer closer colors # TODO: make this a parameter?
        color_distances = torch.cdist(color_centroids, color_palette, p=2)  # [K, num_colors]
        color_logits = -scaling_factor * color_distances ** 2  # Convert distances to logits
        
        # Sample discrete colors using Gumbel-Softmax
        color_samples = self.gumbel_softmax(color_logits, temperature=temperature, hard=True)  # [K, num_colors]
        
        # Convert samples to RGB values
        discrete_colors = torch.mm(color_samples, color_palette)  # [K, 3]
        
        # Weighted sum based on assignments
        reconstructed = torch.mm(assignments, discrete_colors)  # [N, 3]
        
        # Reshape back to image
        reconstructed = reconstructed.reshape(H, W, C)
        
        return reconstructed

class PatchTrainer(object):
    def __init__(self, batch_size, img_size, device=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize camera parameters
        self.alpha = 0.0  # Can be adjusted for camera sampling
        self.sampler_probs = None  # Can be initialized if needed
        self.azim = None
        self.azim_inds = None
        self.cameras = None
        self.lights = None
        
        # Initialize TPS parameters
        self.DATA_DIR = './data'  # Set your data directory
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        
        # Move meshes to the correct device when loading
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=self.device)
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=self.device)
        
        # Ensure all mesh attributes are on the correct device
        self.faces = self.mesh_tshirt.textures.faces_uvs_padded().to(self.device)
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded().to(self.device)
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0].to(self.device)
        selected_tshirt = torch.cat([torch.arange(27), torch.arange(28, 31), torch.arange(32, 43)])
        self.tshirt_locations_infos = EasyDict({ # sleeves, torso
            'nparts': 3,
            'centers': [[7.5, 0], [-7.5, 0], [0, 0]],
            'Rs': [1.5, 1.5, 15.0],
            'ntfs': [6, 6, 8],
            'ntws': [6, 6, 8],
            'radius_fixed': [[1.0], [1.0], [0.5]],
            'radius_wrap': [[0.5], [0.5], [1.0]],
            'signs': [-1, -1, 1],
            'selected': selected_tshirt,
        })
        self.initialize_tps2d() # initialize tps2d

        self.tps2d_range_t = 50.0
        self.tps2d_range_r = 0.1

        self.model = YOLOv3Darknet().eval().to(self.device)
        self.model.load_darknet_weights('arch/weights/yolov3.weights')
        
        # Initialize patch transformer
        self.patch_transformer = PatchTransformer().to(self.device)
        self.prob_extractor = YOLOv3MaxProbExtractor(0, 80, self.model, self.img_size).to(device) # NOTE: yolov3 model is hardcoded

    def get_background_img_loader(self, img_dir, shuffle=True):
        loader = torch.utils.data.DataLoader(InriaDataset(img_dir, self.img_size, shuffle=shuffle),
                                           self.batch_size,
                                           shuffle=True,
                                           num_workers=4)
        return loader

    def sample_cameras(self, theta=None, elev=None):
        if theta is not None:
            if isinstance(theta, float) or isinstance(theta, int):
                self.azim = torch.zeros(self.batch_size).fill_(theta)
            elif isinstance(theta, torch.Tensor):
                self.azim = theta.clone()
            elif isinstance(theta, np.ndarray):
                self.azip = torch.from_numpy(theta)
            else:
                raise ValueError
        else:
            if self.alpha > 0:
                exp = (self.alpha * self.sampler_probs).softmax(0)
                azim = torch.multinomial(exp, self.batch_size, replacement=True)
                self.azim_inds = azim
                azim = azim.to(exp)
                self.azim = (azim + azim.new(size=azim.shape).uniform_() - 0.5) * 360 / len(exp)
            else:
                self.azim_inds = None
                self.azim = (torch.zeros(self.batch_size).uniform_() - 0.5) * 360
        if elev is not None:
            elev = torch.zeros(self.batch_size).fill_(elev)
        else:
            elev = 10 + 8 * torch.zeros(self.batch_size).uniform_(-1, 1)
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=self.azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)
        return

    def sample_lights(self, r=None):
        if r is None:
            r = np.random.rand()
        theta = np.random.rand() * 2 * math.pi
        if r < 0.33:
            self.lights = AmbientLights(device=self.device)
        elif r < 0.67:
            self.lights = DirectionalLights(device=self.device, direction=[[np.sin(theta), 0.0, np.cos(theta)]])
        else:
            self.lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])
        return

    def initialize_tps2d(self):
        if not self.DATA_DIR:
            raise ValueError("DATA_DIR must be set before initializing TPS2D")
            
        locations_tshirt_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/tshirt_join/projections/part_all_2p5.pt'), 
                                        map_location='cpu').to(self.device)
        self.infos_tshirt = mesh_utils.get_map_kernel(locations_tshirt_ori, self.faces_uvs_tshirt)

        target_control_points = p3dmd.get_points(self.tshirt_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_tshirt = TPSGridGen(None, target_control_points, locations_tshirt_ori.cpu())
        tps2d_tshirt.to(self.device)
        self.tps2d_tshirt = tps2d_tshirt

    def adversarial_loss(self, reconstructed_image, target_image, model_output, gt):
        """
        Compute adversarial loss combining reconstruction and detection objectives
        Args:
            reconstructed_image: reconstructed image from SLIC
            target_image: original image
            model_output: output from detection model
            gt: ground truth bounding boxes
        """
        # Reconstruction loss (L2 distance)
        recon_loss = F.mse_loss(reconstructed_image, target_image)
        
        # Detection loss (using prob_extractor like in nopants_train.py)
        det_loss, max_prob_list = self.prob_extractor(model_output, gt, loss_type='max_iou', iou_thresh=0.01)
        
        # Total loss (you can adjust these weights)
        total_loss = recon_loss + det_loss
        
        return total_loss, max_prob_list

    def synthesis_image(self, img_batch, use_tps2d=True, use_tps3d=True):
        # Add batch dimension if it's missing
        # TODO: THIS SHOULD BE [8,3,416,416]
        # TODO: ALSO WE SHOULD INIT STUFF FOR TPS3D
        print("img_batch shape in SYNTHESIS IMAGE: ", img_batch.shape)
        if len(img_batch.shape) == 3:
            img_batch = img_batch.unsqueeze(0)
            print("img_batch shape after unsqueeze: ", img_batch.shape)
        if use_tps2d:
            # tps_2d
            source_control_points_tshirt = p3dmd.get_points(self.tshirt_locations_infos, torch.pi / 180 * self.tps2d_range_t, self.tps2d_range_r,
                                                            bs=self.batch_size, random=True)
            locations_tshirt = self.tps2d_tshirt(source_control_points_tshirt.to(self.device))
        else:
            locations_tshirt = None

        if use_tps3d:
            # tps_3d
            source_coordinate = self.tps3d.tps_mesh(max_range=self.max_range, batch_size=self.batch_size).view(-1, 3)
        else:
            source_coordinate = None
        
        # render images
        images_predicted = p3dmd.view_mesh_wrapped([self.mesh_man, self.mesh_tshirt],
                                                   [None, locations_tshirt],
                                                   [None, self.infos_tshirt], source_coordinate,
                                                   cameras=self.cameras, lights=self.lights, image_size=800, fov=45,
                                                   max_faces_per_bin=30000, faces_per_pixel=3)
        adv_batch = images_predicted.permute(0, 3, 1, 2)
        
        # Transform patch onto human using PatchTransformer
        p_img_batch, gt = self.patch_transformer(img_batch, adv_batch)
        
        return p_img_batch, gt

    def train_adversarial_slic(self, image_path, num_epochs=100):
        """
        Train SLIC parameters adversarially
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to model's expected input size
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image).float() / 255.0
        image = image.to(self.device)

        # Initialize data loader for background scenes
        train_loader = self.get_background_img_loader('./data/background')
        
        # Initialize adversarial SLIC
        adv_slic = SLICGenerator(image).to(self.device)
        optimizer = torch.optim.Adam(adv_slic.parameters(), lr=0.01)
        
        et0 = time.time()
        for epoch in tqdm(range(num_epochs)):
            print('######################################')
            ep_det_loss = 0
            ep_loss = 0
            ep_mean_prob = 0
            ep_tv_loss = 0
            eff_count = 0  # record how many images are actually used in training
            
            for i_batch, img_batch in enumerate(train_loader):
                img_batch = img_batch.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Sample different camera angles for robustness
                if i_batch % 20 == 0:
                    self.sample_cameras()
                    self.sample_lights()
                
                # Generate adversarial texture using SLIC
                centroids, assignments = adv_slic(img_batch)
                
                reconstructed_image = adv_slic.reconstruct(img_batch.shape)
                
                # Apply reconstructed texture to human model
                p_img_batch, gt = self.synthesis_image(reconstructed_image)
                
                # Get model predictions
                output = self.model(p_img_batch)
                
                # Compute detection loss
                try:
                    det_loss, max_prob_list = self.prob_extractor(output, gt, loss_type='max_iou', iou_thresh=0.01)
                    eff_count += 1
                except RuntimeError:  # current batch has no bbox detected
                    continue
                
                # Total loss (simplified from original)
                loss = det_loss
                
                # Add TV regularization if needed
                tv_loss = torch.tensor([0])
                # if args.tv_loss > 0:
                #     tv_loss = total_variation(reconstructed_image)
                #     loss += tv_loss * args.tv_loss
                
                # Update statistics
                ep_mean_prob += max_prob_list.mean().item()
                ep_det_loss += det_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Optional: Visualize progress
                if i_batch % 10 == 0:
                    print(f'Batch {i_batch}, Loss: {loss.item():.4f}')
            
            # Epoch statistics
            et1 = time.time()
            ep_det_loss = ep_det_loss / max(eff_count, 1)
            ep_loss = ep_loss / max(eff_count, 1)
            ep_tv_loss = ep_tv_loss / max(eff_count, 1)
            ep_mean_prob = ep_mean_prob / max(eff_count, 1)
            
            print(f'Epoch {epoch}:')
            print(f'Total Loss: {ep_loss:.4f}')
            print(f'Det Loss: {ep_det_loss:.4f}')
            print(f'Mean Prob: {ep_mean_prob:.4f}')
            print(f'TV Loss: {ep_tv_loss:.4f}')
            print(f'Epoch Time: {et1 - et0:.2f}s')
            
            # Save checkpoints periodically
            if (epoch + 1) % 50 == 0:
                save_path = f'checkpoints/slic_epoch_{epoch}.pth'
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'centroids': adv_slic.centroids,
                    'assignments': adv_slic.assignments,
                    'model_state': adv_slic.state_dict(),
                }, save_path)
            
            et0 = time.time()
        
        return adv_slic

    def test_adversarial_slic(self, adv_slic, conf_thresh=0.01, iou_thresh=0.1):
        """
        Test SLIC adversarial performance
        """
        test_loader = self.get_background_img_loader('./data/background_test')
        total = 0
        positives = []
        
        with torch.no_grad():
            for i_batch, img_batch in tqdm(enumerate(test_loader)):
                img_batch = img_batch.to(self.device)
                
                # Test across different viewing angles
                thetas = np.linspace(-180, 180, 37)
                for theta in thetas:
                    self.sample_cameras(theta=theta)
                    
                    # Generate and apply adversarial texture
                    reconstructed_image = adv_slic.reconstruct(img_batch.shape)
                    p_img_batch, gt = self.synthesis_image(reconstructed_image)
                    
                    # Get model predictions
                    output = self.model(p_img_batch)
                    total += len(p_img_batch)
                    
                    # Process detections
                    output = utils.get_region_boxes_general(output, self.model, conf_thresh=conf_thresh)
                    
                    for i, boxes in enumerate(output):
                        if len(boxes) == 0:
                            positives.append((0.0, False))
                            continue
                            
                        # Convert boxes to proper format and compute IoUs
                        boxes = self.process_boxes(boxes)  # Helper function to format boxes
                        ious = torchvision.ops.box_iou(boxes, gt[i].unsqueeze(0))
                        
                        # Record detection success/failure
                        if (ious > iou_thresh).any():
                            positives.append((boxes[0, 4].item(), True))  # Use confidence of highest-scoring box
                        else:
                            positives.append((0.0, False))
        
        # Compute precision/recall metrics
        positives = sorted(positives, key=lambda x: x[0], reverse=True)
        precision, recall = self.compute_precision_recall(positives, total)
        
        return precision, recall

    @staticmethod
    def process_boxes(boxes):
        """Helper function to process detection boxes into proper format"""
        if len(boxes) == 0:
            return torch.zeros((0, 5))
        
        # Convert from center format to corner format
        w1 = boxes[..., 0] - boxes[..., 2] / 2
        h1 = boxes[..., 1] - boxes[..., 3] / 2
        w2 = boxes[..., 0] + boxes[..., 2] / 2
        h2 = boxes[..., 1] + boxes[..., 3] / 2
        
        processed_boxes = torch.stack([w1, h1, w2, h2, boxes[..., 4]], dim=-1)
        return processed_boxes

    @staticmethod
    def compute_precision_recall(positives, total):
        """Helper function to compute precision/recall from detection results"""
        tp_counter = fp_counter = 0
        precision = []
        recall = []
        
        for conf, is_true in positives:
            if is_true:
                tp_counter += 1
            else:
                fp_counter += 1
                
            recall.append(tp_counter / total)
            precision.append(tp_counter / (tp_counter + fp_counter) if tp_counter > 0 else 0.0)
        
        return precision, recall

if __name__ == "__main__":
    
    # Initialize trainer
    trainer = PatchTrainer(batch_size=8, img_size=416)
    # Train SLIC
    image_path = "lighthouse_img.png"
    adv_slic = trainer.train_adversarial_slic(image_path, num_epochs=50)
    
    # Test SLIC
    precision, recall = trainer.test_adversarial_slic(adv_slic)

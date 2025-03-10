"""
Training code for Adversarial patch training
"""

#import patch_config
import sys
import os

import time
from datetime import datetime
import argparse
import numpy as np
import scipy
import scipy.interpolate
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from easydict import EasyDict
from PIL import Image

from skimage.io import imread
from skimage.util import img_as_float

from generator import *
from load_data import *
from tps import *
from sample_uniformly_per_color import *
# from transformers import DeformableDetrForObjectDetection
from skimage.metrics import structural_similarity

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import parameter
from torch.autograd import Variable, Function
from torchvision import transforms
import torchvision
# from tensorboardX import SummaryWriter
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
import wandb

# add path for demo utils functions 
sys.path.append(os.path.abspath(''))


from arch.yolov3_models import YOLOv3Darknet
from yolo2.darknet import Darknet
from color_util import *
from train_util import *
import pytorch3d_modify as p3dmd
import mesh_utils as MU


class PatchTrainer(object):
    def __init__(self, args):
        self.args = args
        if args.device is not None:
            device = torch.device(args.device)
            # torch.cuda.set_device(device)
        else:
            device = None
        self.device = device
        self.img_size = 416
        self.DATA_DIR = "./data"

        if args.arch == "rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        elif args.arch == "yolov3":
            self.model = YOLOv3Darknet().eval().to(device)
            self.model.load_darknet_weights('arch/weights/yolov3.weights')
        elif args.arch == "detr":
            self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).eval().to(
                device)
        # elif args.arch == "deformable-detr":
        #     self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").eval().to(device)
        elif args.arch == "yolov2":
            self.model = Darknet('yolo2/cfg/yolov2.cfg').eval().to(device)
            self.model.load_weights('yolo2/yolov2.weights')
        elif args.arch == "mask_rcnn":
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        else:
            raise NotImplementedError

        for p in self.model.parameters():
            p.requires_grad = False

        self.batch_size = args.batch_size

        self.patch_transformer = PatchTransformer().to(device)
        if args.arch == "rcnn":
            self.prob_extractor = MaxProbExtractor(0, 80).to(device)
        elif args.arch == "yolov3":
            self.prob_extractor = YOLOv3MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "detr":
            self.prob_extractor = DetrMaxProbExtractor(0, 80, self.img_size).to(device)
        elif args.arch == "deformable-detr":
            self.prob_extractor = DeformableDetrProbExtractor(0,80,self.img_size).to(device)
        self.tv_loss = TotalVariation()

        self.alpha = args.alpha
        self.azim = torch.zeros(self.batch_size)
        self.blend_params = None

        self.sampler_probs = torch.ones([36]).to(device)
        self.loss_history = torch.ones(36).to(device)
        self.num_history = torch.ones(36).to(device)

        self.train_loader = self.get_loader('./data/background', True)
        self.test_loader = self.get_loader('./data/background_test', True)

        self.epoch_length = len(self.train_loader)
        print(f'One training epoch has {len(self.train_loader.dataset)} images')
        print(f'One test epoch has {len(self.test_loader.dataset)} images')

        color_transform = ColorTransform('color_transform_dim6.npz')
        self.color_transform = color_transform.to(device)

        self.fig_size_H = 340
        self.fig_size_W = 860

        resolution = 1
        h, w = int(self.fig_size_H / resolution), int(self.fig_size_W / resolution)
        self.h, self.w = h, w

        # Set paths
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")

        self.coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)
        self.colors = torch.load("data/camouflage4.pth").float().to(device)
        # print(self.colors)
        self.colors = torch.tensor([
            [194, 192, 172],
            [157, 178, 194],
            [106, 115, 84],
            [83, 100, 116],
            [41, 62, 85],
            [46, 114, 75]]).float().to(device)
        self.colors = torch.div(self.colors, 255.)
        num_colors = self.colors.shape[0]

        """
        random_seeds = [
            [[14, 67],[34, 58],[50, 60],[23, 6],[61, 13],[90, 8],[106, 19],[123, 11],[166, 51],[185, 58]],
            [[2, 19],[10, 1],[51, 3],[76, 21],[99, 12],[141, 25],[138, 1],[185, 3],[190, 25],[211, 10]],
            [[16, 13],[10, 48],[12, 76],[62, 44],[76, 68],[95, 76],[89, 29],[128, 44],[149, 79],[203, 52]],
            [[16, 28],[56, 78],[82, 82],[108, 48],[105, 76],[138, 29],[170, 40],[191, 40],[177, 77],[214, 40]],
            [[4, 32],[35, 33],[63, 32],[75, 81],[100, 31],[130, 32],[162, 3],[153, 35],[168, 83],[188, 32]],
            [[0, 49],[27, 45],[71, 35],[85, 38],[106, 39],[119, 39],[135, 39],[176, 40],[169, 43],[205, 42]]
        ]
        for i, r in enumerate(random_seeds):
            np.random.shuffle(r)
        random_seeds = np.array(random_seeds).astype(np.float32)
        random_seeds = np.random.rand(*random_seeds.shape) + random_seeds
        random_seeds[:,:,0] = random_seeds[:,:,0] / self.w
        random_seeds[:,:,1] = random_seeds[:,:,1] / self.h
        random_seeds = np.clip(random_seeds, 0, 1)
        # self.tshirt_point = torch.rand([num_colors, args.num_points_tshirt, 2], requires_grad=True, device=device)
        # np.save("rand.npy", self.tshirt_point.cpu().detach().numpy())
        self.tshirt_point = torch.tensor(random_seeds, requires_grad=True, device=device)
        # np.save("ours.npy", random_seeds)
        # self.tshirt_point = torch.full([num_colors, args.num_points_tshirt, 3], fill_value=0.7, requires_grad=True, device=device)
        """

        points, colors = sample_n_per_color("road.png", num_colors, args.num_points_tshirt)
        self.colors =  torch.tensor(colors).float().to(device)
        self.colors = torch.div(self.colors, 255.)
        num_colors = self.colors.shape[0]
        self.tshirt_point = torch.tensor(points, requires_grad=True, device=device)
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)

        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]

        # self.ref_image = img_as_float(imread("test.png"))
        self.ref_image = np.array(Image.open("road.png").resize((self.fig_size_W, self.fig_size_H)))

        self.optimizer = torch.optim.Adam([self.tshirt_point], args.lr)

        if args.seed_type in ['fixed', 'random']:
            self.seeds_tshirt = torch.zeros(size=[h, w, num_colors], device=device).uniform_()
            self.optimizer_seed = torch.optim.SGD([torch.zeros(1, device=device).requires_grad_()], lr=args.lr_seed)
        else:
            self.seeds_tshirt_train = torch.zeros(size=[h, w, num_colors], device=device).uniform_(args.clamp_shift,
                                                                                              1 - args.clamp_shift).requires_grad_()  # NOTE when not fixed we use uniform

            self.seeds_tshirt_fixed = torch.zeros(size=[h, w, num_colors], device=device).uniform_()

            if args.seed_opt == 'sgd':
                self.optimizer_seed = torch.optim.SGD([self.seeds_tshirt_train], lr=args.lr_seed)
            elif args.seed_opt == 'adam':
                self.optimizer_seed = torch.optim.Adam([self.seeds_tshirt_train], lr=args.lr_seed)
            else:
                raise ValueError

        k = 3
        k2 = k * k
        self.camouflage_kernel = nn.Conv2d(num_colors, num_colors, k, 1, int(k / 2)).to(device)
        self.camouflage_kernel.weight.data.fill_(0)
        self.camouflage_kernel.bias.data.fill_(0)
        for i in range(num_colors):
            self.camouflage_kernel.weight[i, i, :, :].data.fill_(1 / k2)

        self.expand_kernel = nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
        self.expand_kernel.weight.data.fill_(0)
        self.expand_kernel.bias.data.fill_(0)
        for i in range(3):
            self.expand_kernel.weight[i, i, :, :].data.fill_(1)

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

        self.initialize_tps2d()
        self.initialize_tps3d()

    def get_loader(self, img_dir, shuffle=True):
        loader = torch.utils.data.DataLoader(InriaDataset(img_dir, self.img_size, shuffle=shuffle),
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=4)
        return loader

    def init_tensorboard(self, name=None):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        print(time_str)
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        fname = self.args.save_path.split('/')[-1]
        return SummaryWriter(f'runs_new/{TIMESTAMP}_{fname}')

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
        locations_tshirt_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/tshirt_join/projections/part_all_2p5.pt'), map_location='cpu').to(self.device)
        self.infos_tshirt = MU.get_map_kernel(locations_tshirt_ori, self.faces_uvs_tshirt)

        target_control_points = p3dmd.get_points(self.tshirt_locations_infos, wrap=False).squeeze(0).cpu()
        tps2d_tshirt = TPSGridGen(None, target_control_points, locations_tshirt_ori.cpu())
        tps2d_tshirt.to(self.device)
        self.tps2d_tshirt = tps2d_tshirt

        return

    def initialize_tps3d(self):
        xmin, ymin, zmin = (-0.28170400857925415, -0.7323740124702454, -0.15313300490379333)
        xmax, ymax, zmax = (0.28170400857925415, 0.5564370155334473, 0.0938199982047081)
        xnum, ynum, znum = [5, 8, 5]
        max_range = (torch.Tensor([xmax, ymax, zmax]) - torch.Tensor([xmin, ymin, zmin])) / torch.Tensor(
            [xnum, ynum, znum])
        self.max_range = (max_range * self.args.tps3d_range).tolist()
        target_control_points = torch.tensor(list(itertools.product(
            torch.linspace(xmin, xmax, xnum),
            torch.linspace(ymin, ymax, ynum),
            torch.linspace(zmin, zmax, znum),
        )))
        mesh = MU.join_meshes([self.mesh_man, self.mesh_tshirt])

        tps3d = TPSGridGen(None, target_control_points, mesh.verts_packed().cpu())
        tps3d.to(self.device)
        self.tps3d = tps3d
        return

    def synthesis_image(self, img_batch, use_tps2d=True, use_tps3d=True):
        if use_tps2d:
            # tps_2d
            source_control_points_tshirt = p3dmd.get_points(self.tshirt_locations_infos, torch.pi / 180 * self.args.tps2d_range_t, self.args.tps2d_range_r,
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
        p_img_batch, gt = self.patch_transformer(img_batch, adv_batch)
        return p_img_batch, gt

    def update_mesh(self, tau=0.3, type='gumbel'):
        # camouflage:
        prob_map = prob_fix_color(self.tshirt_point, self.coordinates, self.colors, self.h, self.w, blur=self.args.blur).unsqueeze(0)
        prob_map = self.camouflage_kernel(prob_map)
        prob_map = prob_map.squeeze(0).permute(1, 2, 0)

        gb_tshirt = -(-(self.seeds_tshirt + 1e-20).log() + 1e-20).log()

        tex = gumbel_color_fix_seed(prob_map, gb_tshirt, self.colors, tau=tau, type=type)

        tex = self.expand_kernel(self.color_transform(tex.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        Image.fromarray((tex[0].detach().cpu().numpy()*255).astype(np.uint8)).show()
        print(tex.mean())
        # tex = plt.imread("/home/cynthia/uw/Adversarial_camou/xp.png")[None,:]
        self.mesh_tshirt.textures = TexturesUV(maps=tex, faces_uvs=self.faces, verts_uvs=self.verts_uv)

        return tex

    def load_weights(self, save_path, epoch):
        path = save_path + '/' + str(epoch) + '_circle_epoch.pth'
        self.tshirt_point.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '/' + str(epoch) + '_color_epoch.pth'
        self.colors.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '/' + str(epoch) + '_seed_tshirt_epoch.pth'
        self.seeds_tshirt = torch.load(path, map_location='cpu').to(self.device)

        if self.args.seed_type in ['variable', 'langevin']:
            path = save_path + '/' + str(epoch) + '_seed_tshirt_train_epoch.pth'
            self.seeds_tshirt_train.data = torch.load(path, map_location='cpu').to(self.device)

            path = save_path + '/' + str(epoch) + '_seed_tshirt_fixed_epoch.pth'
            self.seeds_tshirt_fixed.data = torch.load(path, map_location='cpu').to(self.device)

        path = save_path + '/' + str(epoch) + 'info.npz'
        if os.path.exists(path):
            x = np.load(path)
            self.loss_history = torch.from_numpy(x['loss_history']).to(self.device)
            self.num_history = torch.from_numpy(x['num_history']).to(self.device)

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        # self.writer = self.init_tensorboard()
        args = self.args

        wandb.define_metric("epoch", step_metric="epoch")
        wandb.define_metric("epoch_loss", step_metric="epoch")
        wandb.define_metric("detection_loss", step_metric="epoch")
        wandb.define_metric("mean_prob", step_metric="epoch")
        wandb.define_metric("tv_loss", step_metric="epoch")
        wandb.define_metric("ctrl_loss", step_metric="epoch")
        wandb.define_metric("seed_loss", step_metric="epoch")
        wandb.define_metric("epoch_time", step_metric="epoch")

        et0 = time.time()
        checkpoints = args.checkpoints
        if checkpoints > 0:
            self.load_weights(args.save_path, checkpoints - 1)

        torch.autograd.set_detect_anomaly(True)
        for epoch in tqdm(range(checkpoints, args.nepoch)):
            print('######################################')
            ep_det_loss = 0
            ep_loss = 0
            ep_mean_prob = 0
            ep_tv_loss = 0
            ep_ctrl_loss = 0
            ep_seed_loss = 0
            # ep_ssim_loss = 0
            ep_log_likelihood = 0
            ep_sim_loss = 0
            eff_count = 0  # record how many images in this epoch are really in training so that we can calculate accurate loss
            self.sampler_probs = self.loss_history / self.num_history
            if epoch % 100 == 0:
                print(self.sampler_probs)
            self.loss_history = self.loss_history / 2 + 1e-5
            self.num_history = self.num_history / 2 + 1e-5
            if epoch % 100 == 99:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / args.lr_decay
                self.optimizer_seed.param_groups[0]['lr'] = self.optimizer_seed.param_groups[0]['lr'] / args.lr_decay_seed

            if args.anneal:
                tau = np.exp(-(epoch + 1) / args.nepoch * args.anneal_alpha) * args.anneal_init
            else:
                tau = 0.3

            for i_batch, img_batch in enumerate(self.train_loader):
                img_batch = img_batch.to(self.device)
                t0 = time.time()
                # AG step
                self.optimizer.zero_grad()
                self.optimizer_seed.zero_grad()
                if i_batch % 20 == 0:
                    self.sample_cameras()
                    self.sample_lights()

                if args.seed_type in ['variable', 'langevin']:
                    self.seeds_tshirt = args.seed_ratio * self.seeds_tshirt_train + (1 - args.seed_ratio) * self.seeds_tshirt_fixed

                tex = self.update_mesh(tau=tau)
                # Image.fromarray(img_batch[0].cpu().numpy())
                p_img_batch, gt = self.synthesis_image(img_batch, not args.disable_tps2d, not args.disable_tps3d)
                t1 = time.time()
                normalize = True
                if self.args.arch == "deformable-detr" and normalize:
                    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                    p_img_batch = normalize(p_img_batch)
                output = self.model(p_img_batch)

                t2 = time.time()
                try:
                    det_loss, max_prob_list = self.prob_extractor(output, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                    eff_count += 1
                except RuntimeError as e:  # current batch of imgs have no bbox be detected
                    print("ERROR", e)
                    continue
                t3 = time.time()
                if self.azim_inds is not None:
                    self.loss_history.index_put_([self.azim_inds], max_prob_list.detach(), accumulate=True)
                    self.num_history.index_put_([self.azim_inds], torch.ones_like(max_prob_list), accumulate=True)
                loss = 0
                tv_loss = torch.tensor([0])
                loss += 0.1*det_loss
                if args.tv_loss > 0:
                    tv_loss = self.tv_loss(tex)
                    loss += tv_loss * args.tv_loss

                loss_c = ctrl_loss(self.tshirt_point, self.fig_size_H, self.fig_size_W)
                loss += args.ctrl * loss_c

                # if args.loss_ssim != 0:
                #     loss_ssim = 1-structural_similarity(tex[0].detach().cpu().numpy(), self.ref_image, channel_axis=2, data_range=self.ref_image.max()-self.ref_image.min())
                #     loss += loss_ssim

                if args.cdist != 0:
                    loss_seed = args.cdist * reg_dist(self.seeds_tshirt_train.flatten(), sample_num=args.rd_num)
                    loss += loss_seed
                else:
                    loss_seed = torch.zeros([], device=self.device)

                ep_mean_prob += max_prob_list.mean().item()
                ep_ctrl_loss += loss_c.item()
                ep_det_loss += det_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_seed_loss += loss_seed.item()
                # ep_ssim_loss += loss_ssim.item()
                ep_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if args.seed_type == 'random':
                    self.seeds_tshirt.uniform_()
                elif args.seed_type != 'fixed':
                    self.seeds_tshirt_train.grad /= args.seed_temp
                    self.optimizer_seed.step()
                    self.seeds_tshirt_train.data.clamp_(args.clamp_shift, 1 - args.clamp_shift)
                    if args.seed_type == 'langevin':
                        beta = np.sqrt(2 * self.optimizer_seed.param_groups[0]['lr'])
                        for s in [self.seeds_tshirt_train]:
                            # assert beta and clamp_shift are both small
                            raw = s + s.new(s.shape).normal_() * beta
                            s.data = raw.clamp(args.clamp_shift, 1 - args.clamp_shift) * 2 - raw
                            s.data.clamp_(args.clamp_shift, 1 - args.clamp_shift)
                t4 = time.time()
                self.tshirt_point.data = self.tshirt_point.data.clamp(0, 1)
                self.colors.data = self.colors.data.clamp(0, 1)

                if i_batch % 10 == 0:
                    iteration = self.epoch_length * epoch + i_batch
                    # self.writer.add_scalar('batch/total_loss', loss.detach().cpu().numpy(), iteration)
                    # self.writer.add_scalar('batch/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                    # self.writer.add_scalar('batch/det_loss', det_loss.detach().cpu().numpy(), iteration)
                    # self.writer.add_scalar('batch/ctrl_loss', loss_c.detach().cpu().numpy(), iteration)
                    # self.writer.add_scalar('batch/loss_seed', loss_seed.detach().cpu().numpy(), iteration)

            et1 = time.time()
            ep_det_loss = ep_det_loss / eff_count
            ep_loss = ep_loss / eff_count
            ep_tv_loss = ep_tv_loss / eff_count
            ep_ctrl_loss = ep_ctrl_loss / eff_count
            ep_mean_prob = ep_mean_prob / eff_count
            ep_seed_loss = ep_seed_loss / eff_count
            # ep_ssim_loss = ep_ssim_loss / eff_count
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print(' MEAN PROB: ', ep_mean_prob)
                print('   TV LOSS: ', ep_tv_loss)
                print(' CTRL LOSS: ', ep_ctrl_loss)
                print(' SEED LOSS: ', ep_seed_loss)
                # print(' SSIM LOSS: ', ep_ssim_loss)
                print('EPOCH TIME: ', et1 - et0)
                wandb.log({
                    "epoch": epoch,
                    "epoch_loss": ep_loss,
                    "detection_loss": ep_det_loss,
                    "mean_prob": ep_mean_prob,
                    "tv_loss": ep_tv_loss,
                    "ctrl_loss": ep_ctrl_loss,
                    "seed_loss": ep_seed_loss,
                    "epoch_time": et1 - et0,
                    "synthesized_texture": wandb.Image(
                        np.array(255*tex.squeeze(0).detach().cpu()).astype('uint8'),
                        caption=f"Texture at Epoch {epoch}"
                    ),
                    "rendered_image": wandb.Image(
                        np.array(255*p_img_batch[0].permute(1,2,0).detach().cpu()).astype('uint8'),
                        caption=f"Rendered at Epoch {epoch}"
                    )
                })
                # if epoch % 2 == 0:
                    # plt.imshow(tex[0].detach().cpu().numpy())
                    # plt.pause(0.1)
                    # plt.cla()
                    # print(tex[0].max(), tex[0].min(), tex[0].shape)
                    # plt.imsave("intermediate.png", tex[0].detach().clamp(0,1).cpu().numpy())
                    # _ = input()

                # self.writer.add_scalar('epoch/total_loss', ep_loss, epoch)
                # self.writer.add_scalar('epoch/tv_loss', ep_tv_loss, epoch)
                # self.writer.add_scalar('epoch/det_loss', ep_det_loss, epoch)
                # self.writer.add_scalar('epoch/ctrl_loss', ep_ctrl_loss, epoch)
                # self.writer.add_scalar('epoch/seed_loss', ep_seed_loss, epoch)
                # self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
            et0 = time.time()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                tex_img = Image.fromarray(np.array(255*tex.squeeze(0).detach().cpu()).astype('uint8'))
                tex_img.save(f"epoch_{epoch}.png")
                # tex_img.show()
                # self.writer.add_figure('maps_tshirt', fig, epoch)

            if (epoch + 1) % 50 == 0:
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                path = args.save_path + '/' + str(epoch) + '_circle_epoch.pth'
                torch.save(self.tshirt_point, path)
                path = args.save_path + '/' + str(epoch) + '_color_epoch.pth'
                torch.save(self.colors, path)
                path = args.save_path + '/' + str(epoch) + '_seed_tshirt_epoch.pth'
                torch.save(self.seeds_tshirt, path)

                if args.seed_type in ['variable', 'langevin']:
                    path = args.save_path + '/' + str(epoch) + '_seed_tshirt_train_epoch.pth'
                    torch.save(self.seeds_tshirt_train, path)

                    path = args.save_path + '/' + str(epoch) + '_seed_tshirt_fixed_epoch.pth'
                    torch.save(self.seeds_tshirt_fixed, path)

                path = args.save_path + '/' + str(epoch) + 'info.npz'
                np.savez(path, loss_history=self.loss_history.cpu().numpy(), num_history=self.num_history.cpu().numpy(), azim=self.azim.cpu().numpy())

            if (epoch + 1) % 300 == 0:
                self.update_mesh(type='determinate')
                for iou_thresh in [0.01, 0.1, 0.3, 0.5]:
                    precision, recall, avg, confs, thetas = self.test(conf_thresh=0.01, iou_thresh=iou_thresh, angle_sample=37, use_tps2d=not args.disable_test_tps2d, use_tps3d=not args.disable_test_tps3d, mode=args.test_mode)
                    info = [precision, recall, avg, confs]
                    path = args.save_path + '/' + str(epoch) + 'test_results_tps'
                    path = path + '_iou' + str(iou_thresh).replace('.', '') + '_' + args.test_mode
                    path = path + '.npz'
                    np.savez(path, thetas=thetas, info=info)


    def test(self, conf_thresh, iou_thresh, num_of_samples=100, angle_sample=37, use_tps2d=True, use_tps3d=True, mode='person'):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        print(f'One test epoch has {len(self.test_loader.dataset)} images')

        thetas_list = np.linspace(-180, 180, angle_sample)
        confs = [[] for i in range(angle_sample)]
        self.sample_lights(r=0.1)

        total = 0.
        positives = []
        et0 = time.time()
        with torch.no_grad():
            j = 0
            for i_batch, img_batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), position=0):
                img_batch = img_batch.to(self.device)
                for it, theta in enumerate(thetas_list):
                    self.sample_cameras(theta=theta)
                    p_img_batch, gt = self.synthesis_image(img_batch, use_tps2d, use_tps3d)

                    normalize = True
                    if self.args.arch == "deformable-detr" and normalize:
                        normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        p_img_batch = normalize(p_img_batch)

                    output = self.model(p_img_batch)
                    total += len(p_img_batch)  # since 1 image only has 1 gt, so the total # gt is just = the total # images
                    pos = []
                    # for i, boxes in enumerate(output):  # for each image
                    conf_thresh = 0.0 if self.args.arch in ['rcnn'] else 0.1
                    person_cls = 0
                    output = utils.get_region_boxes_general(output, self.model, conf_thresh=conf_thresh, name=self.args.arch)

                    for i, boxes in enumerate(output):
                        if len(boxes) == 0:
                            pos.append((0.0, False))
                            continue
                        assert boxes.shape[1] == 7
                        boxes = utils.nms(boxes, nms_thresh=args.test_nms_thresh)
                        w1 = boxes[..., 0] - boxes[..., 2] / 2
                        h1 = boxes[..., 1] - boxes[..., 3] / 2
                        w2 = boxes[..., 0] + boxes[..., 2] / 2
                        h2 = boxes[..., 1] + boxes[..., 3] / 2
                        bboxes = torch.stack([w1, h1, w2, h2], dim=-1)
                        bboxes = bboxes.view(-1, 4).detach() * self.img_size
                        scores = boxes[..., 4]
                        labels = boxes[..., 6]

                        if (len(bboxes) == 0):
                            pos.append((0.0, False))
                            continue
                        scores_ordered, inds = scores.sort(descending=True)
                        scores = scores_ordered
                        bboxes = bboxes[inds]
                        labels = labels[inds]
                        inds_th = scores > conf_thresh
                        scores = scores[inds_th]
                        bboxes = bboxes[inds_th]
                        labels = labels[inds_th]

                        if mode == 'person':
                            inds_label = labels == person_cls
                            scores = scores[inds_label]
                            bboxes = bboxes[inds_label]
                            labels = labels[inds_label]
                        elif mode == 'all':
                            pass
                        else:
                            raise ValueError

                        if (len(bboxes) == 0):
                            pos.append((0.0, False))
                            continue
                        ious = torchvision.ops.box_iou(bboxes.data,
                                                       gt[i].unsqueeze(0))  # get iou of all boxes in this image
                        noids = (ious.squeeze(-1) > iou_thresh).nonzero()
                        if noids.shape[0] == 0:
                            pos.append((0.0, False))
                        else:
                            noid = noids.min()
                            if labels[noid] == person_cls:
                                pos.append((scores[noid].item(), True))
                            else:
                                pos.append((scores[noid].item(), False))
                    positives.extend(pos)
                    confs[it].extend([p[0] if p[1] else 0.0 for p in pos])
       

        print(len(positives) / total) 
        positives = sorted(positives, key=lambda d: d[0], reverse=True)
        confs = np.array(confs)
        tps = []
        fps = []
        tp_counter = 0
        fp_counter = 0
        # all matches in dataset
        print("FOUND ", len(positives), "POSITIVES")
        for pos in positives:
            if pos[1]:
                tp_counter += 1
            else:
                fp_counter += 1
            tps.append(tp_counter)
            fps.append(fp_counter)
        precision = []
        recall = []
        for tp, fp in zip(tps, fps):
            recall.append(tp / total)
            if tp == 0:
                precision.append(0.0)
            else:
                precision.append(tp / (fp + tp))

        if len(precision) > 1 and len(recall) > 1:
            p = np.array(precision)
            r = np.array(recall)
            p_start = p[np.argmin(r)]
            samples = np.linspace(0., 1., num_of_samples)
            interpolated = scipy.interpolate.interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
            avg = sum(interpolated) / len(interpolated)
        elif len(precision) > 0 and len(recall) > 0:
            # 1 point on PR: AP is box between (0,0) and (p,r)
            avg = precision[0] * recall[0]
        else:
            avg = float('nan')

        print(avg)
        return precision, recall, avg, confs, thetas_list


if __name__ == '__main__':
    print('Version 2.0')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--device', default='cuda:1', help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--lr_seed', type=float, default=0.01, help='')
    parser.add_argument('--nepoch', type=int, default=60, help='')
    parser.add_argument('--checkpoints', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--save_path', default='results/', help='')
    parser.add_argument("--alpha", type=float, default=10, help='')
    parser.add_argument("--tv_loss", type=float, default=0, help='')
    parser.add_argument("--lr_decay", type=float, default=2, help='')
    parser.add_argument("--lr_decay_seed", type=float, default=2, help='')
    parser.add_argument("--blur", type=float, default=1, help='')
    parser.add_argument("--like", type=float, default=1, help='')
    parser.add_argument("--ctrl", type=float, default=1, help='')
    parser.add_argument("--num_points_tshirt", type=int, default=60, help='')
    parser.add_argument("--num_points_trouser", type=int, default=60, help='')
    parser.add_argument("--arch", type=str, default="rcnn")
    parser.add_argument("--cdist", type=float, default=0, help='')
    parser.add_argument("--seed_type", default='fixed', help='fixed, random, variable, langevin')
    parser.add_argument("--rd_num", type=int, default=200, help='')
    parser.add_argument("--clamp_shift", type=float, default=0, help='')
    parser.add_argument("--resample_type", default=None, help='')
    parser.add_argument("--seed_temp", type=float, default=1.0, help='')
    parser.add_argument("--seed_opt", default='adam', help='')
    parser.add_argument("--tps2d_range_t", type=float, default=50.0, help='')
    parser.add_argument("--tps2d_range_r", type=float, default=0.1, help='')
    parser.add_argument("--tps3d_range", type=float, default=0.15, help='')
    parser.add_argument("--disable_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--disable_test_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_test_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--seed_ratio", default=1.0, type=float, help='The ratio of trainable part when seed type is variable')
    parser.add_argument("--loss_type", default='max_iou', help='max_iou, max_conf, softplus_max, softplus_sum')
    parser.add_argument("--test", default=False, action='store_true', help='')
    parser.add_argument("--test_iou", type=float, default=0.1, help='')
    parser.add_argument("--test_nms_thresh", type=float, default=1.0, help='')
    parser.add_argument("--test_mode", default='person', help='person, all')
    parser.add_argument("--test_suffix", default='', help='')
    parser.add_argument("--train_iou", type=float, default=0.01, help='')
    parser.add_argument("--anneal", default=False, action='store_true', help='')
    parser.add_argument("--anneal_init", type=float, default=5.0, help='')
    parser.add_argument("--anneal_alpha", type=float, default=3.0, help='')


    args = parser.parse_args()
    assert args.seed_type in ['fixed', 'random', 'variable', 'langevin']

    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Train info:", args)
    trainer = PatchTrainer(args)
    if not args.test:
        config = vars(args)
        config["epochs"] = args.nepoch
        wandb.init(project="invisibility_knit", config=config, id="nopants-" + str(args.nepoch) + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        trainer.train()
    else:
        epoch = args.checkpoints - 1
        trainer.load_weights(args.save_path, epoch)
        trainer.update_mesh(type='determinate')
        precision, recall, avg, confs, thetas = trainer.test(conf_thresh=0.01, iou_thresh=args.test_iou, angle_sample=37, use_tps2d=not args.disable_test_tps2d, use_tps3d=not args.disable_test_tps3d, mode=args.test_mode)
        info = [precision, recall, avg, confs]
        path = args.save_path + '/' + str(epoch) + 'test_results_tps'
        path = path + '_iou' + str(args.test_iou).replace('.', '') + '_' + args.test_mode + args.test_suffix
        path = path + '.npz'
        np.savez(path, thetas=thetas, info=info)


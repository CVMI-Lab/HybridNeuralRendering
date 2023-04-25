import math
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..helpers.networks import init_seq, positional_encoding
from utils.spherical import SphericalHarm_table as SphericalHarm
from ..helpers.geometrics import compute_world2local_dist
from .attention import AttentionBlock


def drop_patch_rays(patch_size, patch_num, drop_ratio):
    drop_ray_flag = np.zeros((patch_size * patch_num, patch_size * patch_num))
    drop_patch_num = int(patch_num * patch_num * drop_ratio)
    row = drop_patch_num // patch_num
    col = drop_patch_num % patch_num
    drop_ray_flag[0:(row * patch_size), :] = 1
    drop_ray_flag[(row * patch_size):(row * patch_size + patch_size), 0:(col * patch_size)] = 1
    drop_ray_flag_flat = drop_ray_flag.flatten()
    ray_drop_positions = np.where(drop_ray_flag_flat == 1)
    return ray_drop_positions[0]


class PointAggregator(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            '--feature_init_method',
            type=str,
            default="rand",
            help='which agg model to use [feature_interp | graphconv | affine_mix]')

        parser.add_argument(
            '--which_agg_model',
            type=str,
            default="viewmlp",
            help='which agg model to use [viewmlp | nsvfmlp]')

        parser.add_argument(
            '--agg_distance_kernel',
            type=str,
            default="quadric",
            help='which agg model to use [quadric | linear | feat_intrp | harmonic_intrp]')

        parser.add_argument(
            '--sh_degree',
            type=int,
            default=4,
            help='degree of harmonics')

        parser.add_argument(
            '--sh_dist_func',
            type=str,
            default="sh_quadric",
            help='sh_quadric | sh_linear | passfunc')

        parser.add_argument(
            '--sh_act',
            type=str,
            default="sigmoid",
            help='sigmoid | tanh | passfunc')

        parser.add_argument(
            '--agg_axis_weight',
            type=float,
            nargs='+',
            default=None,
            help=
            '(1., 1., 1.)'
        )

        parser.add_argument(
            '--agg_dist_pers',
            type=int,
            default=1,
            help='use pers dist')

        parser.add_argument(
            '--apply_pnt_mask',
            type=int,
            default=1,
            help='use pers dist')

        parser.add_argument(
            '--modulator_concat',
            type=int,
            default=0,
            help='use pers dist')

        parser.add_argument(
            '--agg_intrp_order',
            type=int,
            default=0,
            help='interpolate first and feature mlp 0 | feature mlp then interpolate 1 | feature mlp color then interpolate 2')

        parser.add_argument(
            '--shading_feature_mlp_layer0',
            type=int,
            default=0,
            help='interp to agged features mlp num')

        parser.add_argument(
            '--shading_feature_mlp_layer1',
            type=int,
            default=2,
            help='interp to agged features mlp num')

        parser.add_argument(
            '--shading_feature_mlp_layer2',
            type=int,
            default=0,
            help='interp to agged features mlp num')

        parser.add_argument(
            '--shading_feature_mlp_layer3',
            type=int,
            default=2,
            help='interp to agged features mlp num')

        parser.add_argument(
            '--shading_feature_num',
            type=int,
            default=256,
            help='agged shading feature channel num')

        parser.add_argument(
            '--point_hyper_dim',
            type=int,
            default=256,
            help='agged shading feature channel num')

        parser.add_argument(
            '--shading_alpha_mlp_layer',
            type=int,
            default=1,
            help='agged features to alpha mlp num')

        parser.add_argument(
            '--shading_color_mlp_layer',
            type=int,
            default=1,
            help='agged features to alpha mlp num')

        parser.add_argument(
            '--shading_color_channel_num',
            type=int,
            default=3,
            help='color channel num')

        parser.add_argument(
            '--num_feat_freqs',
            type=int,
            default=0,
            help='color channel num')

        parser.add_argument(
            '--num_hyperfeat_freqs',
            type=int,
            default=0,
            help='color channel num')

        parser.add_argument(
            '--dist_xyz_freq',
            type=int,
            default=2,
            help='color channel num')

        parser.add_argument(
            '--dist_xyz_deno',
            type=float,
            default=0,
            help='color channel num')

        parser.add_argument(
            '--weight_xyz_freq',
            type=int,
            default=2,
            help='color channel num')

        parser.add_argument(
            '--weight_feat_dim',
            type=int,
            default=8,
            help='color channel num')

        parser.add_argument(
            '--agg_weight_norm',
            type=int,
            default=1,
            help='normalize weight, sum as 1')

        parser.add_argument(
            '--view_ori',
            type=int,
            default=0,
            help='0 for pe+3 orignal channels')

        parser.add_argument(
            '--agg_feat_xyz_mode',
            type=str,
            default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--agg_alpha_xyz_mode',
            type=str,
            default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--agg_color_xyz_mode',
            type=str,
            default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--act_type',
            type=str,
            default="ReLU",
            # default="LeakyReLU",
            help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--act_super',
            type=int,
            default=1,
            # default="LeakyReLU",
            help='1 to use softplus and widden sigmoid for last activation')

        parser.add_argument(
            '--use_2D_CNN',
            type=int,
            default=0,
            help='0 not to use 2D CNNs, 1 to use 2D CNN to predict color')

        parser.add_argument(
            '--use_nearest',
            type=int,
            default=0,
            help='0 not to use aux image, >0 to use xxx nearest warp')

        parser.add_argument(
            '--add_idx',
            type=int,
            default=0,
            help='1: use frame idx as another aux input')

        parser.add_argument(
            '--drop_ratio',
            type=float,
            default=0.0,
            help='randomly drop aux information from nearest frames')

        parser.add_argument(
            '--drop_test',
            type=int,
            default=0,
            help='average the features with and without using aux images')

        parser.add_argument(
            '--drop_disturb_range',
            type=int,
            default=0,
            help='0: set positions as 0; > 0: disturb the reprojected positions')

        parser.add_argument(
            '--drop_patch',
            type=int,
            default=0,
            help='1: drop all rays in a patch')

        parser.add_argument(
            '--ray_points',
            type=int,
            default=1,
            help='0: randomly drop query points image feature; 1: drop all query points image features on a ray')

        parser.add_argument(
            '--disable_viewdirs',
            type=int,
            default=0,
            help='1: do not use view directions in rgb predictions')

        parser.add_argument(
            '--learn_residuals',
            type=int,
            default=1,
            help='0: learn features, 1: learn residuals')

        parser.add_argument(
            '--random_position',
            type=int,
            default=1,
            help='0: add random feature before multi-frame aggregation, '
                 '1: add random feature after multi-frame aggregation')

        parser.add_argument(
            '--mixup_mode',
            type=str,
            default='partial',
            help='partial: part of the 3D feature for blending, full: all dimensions for blending')

        parser.add_argument(
            '--feature_guidance',
            type=int,
            default=1,
            help='0: do not use 3D feature to guide the multi-frame aggregation')

        parser.add_argument(
            '--separate_color_decoder',
            type=int,
            default=0,
            help='1: use different decoders to decode point features and hybrid features')

        parser.add_argument(
            '--large_color_final_block',
            type=int,
            default=0,
            help='1: more layers used to predict the color')

        parser.add_argument(
            '--refine_blend',
            type=int,
            default=0,
            help='0: directly blend the aux features. 1: refine the aux feature first, then blend them.')

        parser.add_argument(
            '--dynamic_weight',
            type=int,
            default=0,
            help='1: use predicted weight to blend the 3D feature (1-w) and aggregated aux/image feature (w).')

        parser.add_argument(
            '--search_size',
            type=int,
            default=0,
            help='search nearby positions')

        parser.add_argument(
            '--search_dilation',
            type=int,
            default=0,
            help='dilation/step')

        parser.add_argument(
            '--exp_aggregation',
            type=float,
            default=1.0,
            help='exp(w/??), should > 0')

        parser.add_argument(
            '--tradition_attention',
            type=int,
            default=0,
            help='1: use the traditional attention block')

        parser.add_argument(
            '--frame_level_attention',
            type=int,
            default=0,
            help='1: separately use the traditional attention in one frame, then aggregate multiple frames via weights')

        parser.add_argument(
            '--use_gumbel_softmax',
            type=int,
            default=0,
            help='1: use the gumbel softmax')

        parser.add_argument(
            '--disable_color_feature',
            type=int,
            default=0,
            help='0: use the neural 3D feature')

        parser.add_argument(
            '--use_delta_view',
            type=int,
            default=1,
            help='0: do not use the delta view')

        parser.add_argument(
            '--downweight_blurry_feats',
            type=int,
            default=0,
            help='1: down weight blurry image feats; 0: not to down weight')

        parser.add_argument(
            '--learnable_blur_kernel',
            type=int,
            default=0,
            help='1: learn the blur kernel')

        parser.add_argument(
            '--learnable_blur_kernel_size',
            type=int,
            default=9,
            help='x: the size of learned blur kernel')

        parser.add_argument(
            '--learnable_blur_patch_size',
            type=int,
            default=8,
            help='x: patch size used to predict the blur kernel')

        parser.add_argument(
            '--learnable_blur_kernel_mode',
            type=int,
            default=4,
            help='0: directly predict the blur kernel'
                 '4: blur kernel * predicted weight + regularization kernel * (1 - predicted weight)')

        parser.add_argument(
            '--learnable_blur_kernel_conv',
            type=int,
            default=0,
            help='1: with spatial conv when predicting blur kernels')

        parser.add_argument(
            '--learnable_blur_kernel_norm',
            type=int,
            default=0,
            help='0: /sum(), 1: softmax')

    def __init__(self, opt):
        super(PointAggregator, self).__init__()
        self.act = getattr(nn, opt.act_type, None)
        print("opt.act_type!!!!!!!!!", opt.act_type)
        self.point_hyper_dim=opt.point_hyper_dim if opt.point_hyper_dim < opt.point_features_dim else opt.point_features_dim

        block_init_lst = []
        if opt.agg_distance_kernel == "feat_intrp":
            feat_weight_block = []
            in_channels = 2 * opt.weight_xyz_freq * 3 + opt.weight_feat_dim
            out_channels = int(in_channels / 2)
            for i in range(2):
                feat_weight_block.append(nn.Linear(in_channels, out_channels))
                feat_weight_block.append(self.act(inplace=True))
                in_channels = out_channels
            feat_weight_block.append(nn.Linear(in_channels, 1))
            feat_weight_block.append(nn.Sigmoid())
            self.feat_weight_mlp = nn.Sequential(*feat_weight_block)
            block_init_lst.append(self.feat_weight_mlp)
        elif opt.agg_distance_kernel == "sh_intrp":
            self.shcomp = SphericalHarm(opt.sh_degree)

        self.opt = opt
        self.dist_dim = (4 if self.opt.agg_dist_pers == 30 else 6) if self.opt.agg_dist_pers > 9 else 3
        self.dist_func = getattr(self, opt.agg_distance_kernel, None)
        assert self.dist_func is not None, "InterpAggregator doesn't have disance_kernel {} ".format(opt.agg_distance_kernel)

        self.axis_weight = None if opt.agg_axis_weight is None else torch.as_tensor(opt.agg_axis_weight, dtype=torch.float32, device="cuda")[None, None, None, None, :]

        self.num_freqs = opt.num_pos_freqs if opt.num_pos_freqs > 0 else 0
        self.num_viewdir_freqs = opt.num_viewdir_freqs if opt.num_viewdir_freqs > 0 else 0

        self.pnt_channels = (2 * self.num_freqs * 3) if self.num_freqs > 0 else 3
        self.viewdir_channels = (2 * self.num_viewdir_freqs * 3 + self.opt.view_ori * 3) if self.num_viewdir_freqs > 0 else 3

        self.which_agg_model = opt.which_agg_model.split("_")[0] if opt.which_agg_model.startswith("feathyper") else opt.which_agg_model
        getattr(self, self.which_agg_model+"_init", None)(opt, block_init_lst)

        self.density_super_act = torch.nn.Softplus()
        self.density_act = torch.nn.ReLU()
        self.color_act = torch.nn.Sigmoid()

        self.shading_patch_size = 1

    def raw2out_density(self, raw_density):
        if self.opt.act_super > 0:
            # return self.density_act(raw_density - 1)  # according to mip nerf, to stablelize the training
            return self.density_super_act(raw_density - 1)  # according to mip nerf, to stablelize the training
        else:
            return self.density_act(raw_density)

    def raw2out_color(self, raw_color):
        color = self.color_act(raw_color)
        if self.opt.act_super > 0:
            color = color * (1 + 2 * 0.001) - 0.001  # according to mip nerf, to stablelize the training
        return color

    def viewmlp_init(self, opt, block_init_lst, shading_patch_size=1):
        self.shading_patch_size = shading_patch_size  # enforce each query point to output a pixel, > 1 will generate a patch.
        assert self.shading_patch_size == 1  # support decoding 1 pixel currently
        dist_xyz_dim = self.dist_dim if opt.dist_xyz_freq == 0 else 2 * abs(opt.dist_xyz_freq) * self.dist_dim
        in_channels = opt.point_features_dim + (0 if opt.agg_feat_xyz_mode == "None" else self.pnt_channels) - (opt.weight_feat_dim if opt.agg_distance_kernel in ["feat_intrp", "meta_intrp"] else 0) - (opt.sh_degree ** 2 if opt.agg_distance_kernel == "sh_intrp" else 0) - (7 if opt.agg_distance_kernel == "gau_intrp" else 0)
        in_channels += (2 * opt.num_feat_freqs * in_channels if opt.num_feat_freqs > 0 else 0) + (dist_xyz_dim if opt.agg_intrp_order > 0 else 0)

        if opt.shading_feature_mlp_layer1 > 0:
            out_channels = opt.shading_feature_num
            block1 = []
            for i in range(opt.shading_feature_mlp_layer1):
                block1.append(nn.Linear(in_channels, out_channels))
                block1.append(self.act(inplace=True))
                in_channels = out_channels
            self.block1 = nn.Sequential(*block1)
            block_init_lst.append(self.block1)
        else:
            self.block1 = self.passfunc

        if opt.shading_feature_mlp_layer2 > 0:
            in_channels = in_channels + (0 if opt.agg_feat_xyz_mode == "None" else self.pnt_channels) + (
                dist_xyz_dim if (opt.agg_intrp_order > 0 and opt.num_feat_freqs == 0) else 0)
            out_channels = opt.shading_feature_num
            block2 = []
            for i in range(opt.shading_feature_mlp_layer2):
                block2.append(nn.Linear(in_channels, out_channels))
                block2.append(self.act(inplace=True))
                in_channels = out_channels
            self.block2 = nn.Sequential(*block2)
            block_init_lst.append(self.block2)
        else:
            self.block2 = self.passfunc

        if opt.shading_feature_mlp_layer3 > 0:
            in_channels = in_channels + (3 if "1" in list(opt.point_color_mode) else 0) + (
                4 if "1" in list(opt.point_dir_mode) else 0)
            out_channels = opt.shading_feature_num
            block3 = []
            for i in range(opt.shading_feature_mlp_layer3):
                block3.append(nn.Linear(in_channels, out_channels))
                block3.append(self.act(inplace=True))
                in_channels = out_channels
            self.block3 = nn.Sequential(*block3)
            block_init_lst.append(self.block3)
        else:
            self.block3 = self.passfunc

        alpha_block = []
        in_channels = opt.shading_feature_num + (0 if opt.agg_alpha_xyz_mode == "None" else self.pnt_channels)
        out_channels = int(opt.shading_feature_num / 2)
        for i in range(opt.shading_alpha_mlp_layer - 1):
            alpha_block.append(nn.Linear(in_channels, out_channels))
            alpha_block.append(self.act(inplace=False))
            in_channels = out_channels
        alpha_block.append(nn.Linear(in_channels, 1))
        self.alpha_branch = nn.Sequential(*alpha_block)
        block_init_lst.append(self.alpha_branch)

        # render RGB color from point features
        color_block = []
        in_channels = opt.shading_feature_num + self.viewdir_channels + (
            0 if opt.agg_color_xyz_mode == "None" else self.pnt_channels)
        out_channels = int(opt.shading_feature_num / 2)
        for i in range(opt.shading_color_mlp_layer - 1):
            color_block.append(nn.Linear(in_channels, out_channels))
            color_block.append(self.act(inplace=True))
            in_channels = out_channels
        color_block.append(nn.Linear(in_channels, 3))
        self.color_branch = nn.Sequential(*color_block)
        block_init_lst.append(self.color_branch)

        # color branch of neural 3D representation, render color features
        color_feature_block = []
        in_channels = opt.shading_feature_num + self.viewdir_channels + (
            0 if opt.agg_color_xyz_mode == "None" else self.pnt_channels)
        out_channels = int(opt.shading_feature_num / 2)
        for i in range(opt.shading_color_mlp_layer - 1):
            color_feature_block.append(nn.Linear(in_channels, out_channels))
            color_feature_block.append(self.act(inplace=True))
            in_channels = out_channels
        self.color_feature_branch = nn.Sequential(*color_feature_block)
        block_init_lst.append(self.color_feature_branch)

        # aggregate image-based features, predict blending weights
        if self.opt.tradition_attention:
            expand_times = 2
            point_feat_c = int(opt.shading_feature_num / 2)
            image_feat_c = 3*(1+expand_times+expand_times**2+expand_times**3)
            if self.opt.use_delta_view:
                image_feat_c = image_feat_c + 3
            self.attention_block = AttentionBlock(point_feat_c, image_feat_c, inner_channels=16)  # QKV
        else:
            if not opt.refine_blend:
                expand_times = 2
                aux_merge_weight_block = []
                if opt.feature_guidance:
                    aux_in_channels = 3*(1+expand_times+expand_times**2+expand_times**3) + int(opt.shading_feature_num / 2)
                    aux_out_channels = int(opt.shading_feature_num / 4)
                    if opt.use_delta_view:
                        aux_in_channels = aux_in_channels + 3
                else:
                    aux_in_channels = 2*3*(1+expand_times+expand_times**2+expand_times**3) + 3  # +3 means with delta v.
                    aux_out_channels = int(aux_in_channels / 2)
                for i in range(3):
                    aux_merge_weight_block.append(nn.Linear(aux_in_channels, aux_out_channels))
                    aux_merge_weight_block.append(self.act(inplace=False))
                    aux_in_channels = aux_out_channels
                aux_merge_weight_block.append(nn.Linear(aux_in_channels, 1))
                aux_merge_weight_block.append(torch.nn.Sigmoid())
                self.aux_merge_weight_block = nn.Sequential(*aux_merge_weight_block)
                block_init_lst.append(self.aux_merge_weight_block)
            else:
                raise NotImplementedError

        # branch used to extract image-based features from nearby images.
        expand_times = 2   # expand the channel after down sampling
        aux_block_s1 = []
        aux_in_channels = 3  # RGB image as input
        aux_out_channels = aux_in_channels * expand_times
        if opt.add_idx:
            aux_in_channels = aux_in_channels + 2  # one RGB image + frame idx (cos, sin) as auxiliary input
        aux_block_s1.append(nn.Conv2d(aux_in_channels, aux_out_channels, kernel_size=3, stride=2, padding=1))  # 1/2
        aux_block_s1.append(self.act(inplace=True))
        aux_block_s1.append(nn.Conv2d(aux_out_channels, aux_out_channels, kernel_size=3, stride=1, padding=1))
        aux_block_s1.append(self.act(inplace=True))
        self.aux_block_s1 = nn.Sequential(*aux_block_s1)
        block_init_lst.append(self.aux_block_s1)

        aux_block_s2 = []
        aux_in_channels = aux_out_channels
        aux_out_channels = aux_in_channels * expand_times
        aux_block_s2.append(nn.Conv2d(aux_in_channels, aux_out_channels, kernel_size=3, stride=2, padding=1))  # 1/4
        aux_block_s2.append(self.act(inplace=True))
        aux_block_s2.append(nn.Conv2d(aux_out_channels, aux_out_channels, kernel_size=3, stride=1, padding=1))
        aux_block_s2.append(self.act(inplace=True))
        self.aux_block_s2 = nn.Sequential(*aux_block_s2)
        block_init_lst.append(self.aux_block_s2)

        aux_block_s3 = []
        aux_in_channels = aux_out_channels
        aux_out_channels = aux_in_channels * expand_times
        aux_block_s3.append(nn.Conv2d(aux_in_channels, aux_out_channels, kernel_size=3, stride=2, padding=1))  # 1/8
        aux_block_s3.append(self.act(inplace=True))
        aux_block_s3.append(nn.Conv2d(aux_out_channels, aux_out_channels, kernel_size=3, stride=1, padding=1))
        aux_block_s3.append(self.act(inplace=True))
        self.aux_block_s3 = nn.Sequential(*aux_block_s3)
        block_init_lst.append(self.aux_block_s3)

        # aggregate image-based features and neural 3D features
        color_mixup_block = []
        mixup_in_channels = int(opt.shading_feature_num / 2) + (1 + expand_times ** 1 + expand_times ** 2 + expand_times ** 3) * 3  # down sample: 1/2, 1/4, 1/8
        mixup_out_channels = int(opt.shading_feature_num / 2)

        if opt.mixup_mode == 'partial':
            assert int(opt.shading_feature_num / 2) > (1+expand_times**1+expand_times**2+expand_times**3)*3
            mixup_in_channels = 2*(1+expand_times**1+expand_times**2+expand_times**3)*3  # down sample: 1/2, 1/4, 1/8
            mixup_out_channels = (1+expand_times**1+expand_times**2+expand_times**3)*3

        if opt.learn_residuals:
            color_mixup_block.append(nn.Linear(mixup_in_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
            color_mixup_block.append(nn.Linear(mixup_out_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
            color_mixup_block.append(nn.Linear(mixup_out_channels, mixup_out_channels))
        elif opt.dynamic_weight:
            color_mixup_block.append(nn.Linear(mixup_in_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
            color_mixup_block.append(nn.Linear(mixup_out_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
            color_mixup_block.append(nn.Linear(mixup_out_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
            color_mixup_block.append(nn.Linear(mixup_out_channels, 1))
            color_mixup_block.append(torch.nn.Sigmoid())
        else:
            color_mixup_block.append(nn.Linear(mixup_in_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
            color_mixup_block.append(nn.Linear(mixup_out_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
            color_mixup_block.append(nn.Linear(mixup_out_channels, mixup_out_channels))
            color_mixup_block.append(self.act(inplace=True))
        self.color_mixup_block = nn.Sequential(*color_mixup_block)
        block_init_lst.append(self.color_mixup_block)

        # predict the final color from hybrid feats using MLP
        color_final_block = []
        final_in_channels = int(opt.shading_feature_num / 2)

        if not opt.feature_guidance:
            final_in_channels = (1 + expand_times ** 1 + expand_times ** 2 + expand_times ** 3) * 3

        if opt.large_color_final_block:
            # 2 layers
            color_final_block.append(nn.Linear(final_in_channels, final_in_channels))
            color_final_block.append(self.act(inplace=True))
            color_final_block.append(nn.Linear(final_in_channels, 3 * (self.shading_patch_size ** 2)))
        else:
            # one layer
            color_final_block.append(nn.Linear(final_in_channels, 3 * (self.shading_patch_size ** 2)))
        self.color_final_block = nn.Sequential(*color_final_block)
        block_init_lst.append(self.color_final_block)

        # predict the final color with 3 2D spatial conv and 1 1x1 kernel.
        # since we render patch, we can use tiny CNN to refine the results
        if opt.use_2D_CNN > 0:
            color_final_block_v2 = []
            final_in_channels = int(opt.shading_feature_num / 2)
            color_final_block_v2.append(nn.Conv2d(final_in_channels, final_in_channels, kernel_size=3, stride=1, padding=1))
            color_final_block_v2.append(self.act(inplace=True))
            color_final_block_v2.append(nn.Conv2d(final_in_channels, final_in_channels, kernel_size=3, stride=1, padding=1))
            color_final_block_v2.append(self.act(inplace=True))
            color_final_block_v2.append(nn.Conv2d(final_in_channels, final_in_channels, kernel_size=3, stride=1, padding=1))
            color_final_block_v2.append(self.act(inplace=True))
            color_final_block_v2.append(nn.Conv2d(final_in_channels, 3*(self.shading_patch_size**2), kernel_size=1, stride=1, padding=0))
            self.color_final_block_v2 = nn.Sequential(*color_final_block_v2)
            block_init_lst.append(self.color_final_block_v2)

        # use different decoder to predict query point with and without image feats
        if opt.separate_color_decoder:
            color_final_block_2 = []
            final_in_channels = int(opt.shading_feature_num / 2)

            if opt.large_color_final_block:
                color_final_block_2.append(nn.Linear(final_in_channels, final_in_channels))
                color_final_block_2.append(self.act(inplace=True))
                color_final_block_2.append(nn.Linear(final_in_channels, 3 * (self.shading_patch_size ** 2)))
            else:
                color_final_block_2.append(nn.Linear(final_in_channels, 3 * (self.shading_patch_size ** 2)))
            self.color_final_block_2 = nn.Sequential(*color_final_block_2)
            block_init_lst.append(self.color_final_block_2)

        # predict blurry kernels
        if opt.learnable_blur_kernel:
            learn_blur_kernel_block = []
            blur_in_channels = 2*(opt.learnable_blur_patch_size**2)  # shape of gt patch == shape of rendered patch
            blur_intermediate_channels = 128
            blur_out_channels = opt.learnable_blur_kernel_size**2

            if opt.learnable_blur_kernel_conv:
                # apply conv before MLP
                learn_blur_kernel_conv_block = []
                learn_blur_kernel_conv_block.append(nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=0))
                learn_blur_kernel_conv_block.append(self.act(inplace=True))
                learn_blur_kernel_conv_block.append(nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0))
                learn_blur_kernel_conv_block.append(self.act(inplace=True))
                learn_blur_kernel_conv_block.append(nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0))
                learn_blur_kernel_conv_block.append(self.act(inplace=True))
                learn_blur_kernel_conv_block.append(nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0))
                learn_blur_kernel_conv_block.append(self.act(inplace=True))
                self.learn_blur_kernel_conv_block = nn.Sequential(*learn_blur_kernel_conv_block)
                block_init_lst.append(self.learn_blur_kernel_conv_block)
                blur_in_channels = 8*(opt.learnable_blur_patch_size - 4)*(opt.learnable_blur_patch_size - 4)

            if opt.learnable_blur_kernel_mode == 2 or opt.learnable_blur_kernel_mode == 4:
                blur_out_channels = blur_out_channels + 1  # predict another weight

            # flatten the rendered patch and predict the blur kernel via a small MLP
            learn_blur_kernel_block.append(nn.Linear(blur_in_channels, blur_intermediate_channels))
            learn_blur_kernel_block.append(self.act(inplace=True))
            learn_blur_kernel_block.append(nn.Linear(blur_intermediate_channels, blur_intermediate_channels))
            learn_blur_kernel_block.append(self.act(inplace=True))
            learn_blur_kernel_block.append(nn.Linear(blur_intermediate_channels, blur_intermediate_channels))
            learn_blur_kernel_block.append(self.act(inplace=True))
            learn_blur_kernel_block.append(nn.Linear(blur_intermediate_channels, blur_out_channels))
            learn_blur_kernel_block.append(torch.nn.Sigmoid())
            self.learn_blur_kernel_block = nn.Sequential(*learn_blur_kernel_block)
            block_init_lst.append(self.learn_blur_kernel_block)
        else:
            self.learn_blur_kernel_block = None

        for m in block_init_lst:
            init_seq(m)

    def passfunc(self, input):
        return input

    def trilinear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * 3
        # return B * R * SR * K
        dists = dists * pnt_mask[..., None]
        dists = dists / grid_vox_sz

        #  dist: [1, 797, 40, 8, 3];     pnt_mask: [1, 797, 40, 8]
        # dists = 1 + dists * torch.as_tensor([[1,1,1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=torch.float32, device=dists.device).view(1, 1, 1, 8, 3)

        dists = 1 - torch.abs(dists)

        weights = pnt_mask * dists[..., 0] * dists[..., 1] * dists[..., 2]
        norm_weights = weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)

        # ijk = xyz.astype(np.int32)
        # i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        # V000 = data[i, j, k].astype(np.int32)
        # V100 = data[(i + 1), j, k].astype(np.int32)
        # V010 = data[i, (j + 1), k].astype(np.int32)
        # V001 = data[i, j, (k + 1)].astype(np.int32)
        # V101 = data[(i + 1), j, (k + 1)].astype(np.int32)
        # V011 = data[i, (j + 1), (k + 1)].astype(np.int32)
        # V110 = data[(i + 1), (j + 1), k].astype(np.int32)
        # V111 = data[(i + 1), (j + 1), (k + 1)].astype(np.int32)
        # xyz = xyz - ijk
        # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # Vxyz = (V000 * (1 - x) * (1 - y) * (1 - z)
        #         + V100 * x * (1 - y) * (1 - z) +
        #         + V010 * (1 - x) * y * (1 - z) +
        #         + V001 * (1 - x) * (1 - y) * z +
        #         + V101 * x * (1 - y) * z +
        #         + V011 * (1 - x) * y * z +
        #         + V110 * x * y * (1 - z) +
        #         + V111 * x * y * z)
        return norm_weights, embedding


    def avg(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        weights = pnt_mask * 1.0
        return weights, embedding


    def quadric(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 1] == 1 and axis_weight[..., 2] ==1):
            weights = 1./ torch.clamp(torch.sum(torch.square(dists[..., :3]), dim=-1), min= 1e-8)
        else:
            weights = 1. / torch.clamp(torch.sum(torch.square(dists)* axis_weight, dim=-1), min=1e-8)
        weights = pnt_mask * weights
        return weights, embedding


    def numquadric(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 1] == 1 and axis_weight[..., 2] ==1):
            weights = 1./ torch.clamp(torch.sum(torch.square(dists), dim=-1), min= 1e-8)
        else:
            weights = 1. / torch.clamp(torch.sum(torch.square(dists)* axis_weight, dim=-1), min=1e-8)
        weights = pnt_mask * weights
        return weights, embedding


    def linear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :
            weights = 1. / torch.clamp(torch.norm(dists[..., :3], dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        return weights, embedding


    def numlinear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :
            weights = 1. / torch.clamp(torch.norm(dists, dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        norm_weights = weights / torch.clamp(torch.sum(pnt_mask, dim=-1, keepdim=True), min=1)
        return norm_weights, embedding


    def sigmoid(self, input):
        return torch.sigmoid(input)


    def tanh(self, input):
        return torch.tanh(input)


    def sh_linear(self, dist_norm):
        return 1 / torch.clamp(dist_norm, min=1e-8)


    def sh_quadric(self, dist_norm):
        return 1 / torch.clamp(torch.square(dist_norm), min=1e-8)


    def sh_intrp(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        dist_norm = torch.linalg.norm(dists, dim=-1)
        dist_dirs = dists / torch.clamp(dist_norm[...,None], min=1e-8)
        shall = self.shcomp.sh_all(dist_dirs, filp_dir=False).view(dists.shape[:-1]+(self.shcomp.total_deg ** 2,))
        sh_coefs = embedding[..., :self.shcomp.total_deg ** 2]
        # shall: [1, 816, 24, 32, 16], sh_coefs: [1, 816, 24, 32, 16], pnt_mask: [1, 816, 24, 32]
        # debug: weights = pnt_mask * torch.sum(shall, dim=-1)
        # weights = pnt_mask * torch.sum(shall * getattr(self, self.opt.sh_act, None)(sh_coefs), dim=-1) * getattr(self, self.opt.sh_dist_func, None)(dist_norm)
        weights = pnt_mask * torch.sum(getattr(self, self.opt.sh_act, None)(shall * sh_coefs), dim=-1) * getattr(self, self.opt.sh_dist_func, None)(dist_norm) # changed
        return weights, embedding[..., self.shcomp.total_deg ** 2:]


    def gau_intrp(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # dist: [1, 752, 40, 32, 3]
        B, R, SR, K, _ = dists.shape
        scale = torch.abs(embedding[..., 0]) #
        radii = vsize[2] * 20 * torch.sigmoid(embedding[..., 1:4])
        rotations = torch.clamp(embedding[..., 4:7], max=np.pi / 4, min=-np.pi / 4)
        gau_dist = compute_world2local_dist(dists, radii, rotations)[..., 0]
        # print("gau_dist", gau_dist.shape)
        weights = pnt_mask * scale * torch.exp(-0.5 * torch.sum(torch.square(gau_dist), dim=-1))
        # print("gau_dist", gau_dist.shape, gau_dist[0, 0])
        # print("weights", weights.shape, weights[0, 0, 0])
        return weights, embedding[..., 7:]

    # two-branches viewmlp
    def viewmlp(self, sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, vsize, weight, pnt_mask_flat, pts, viewdirs, total_len, ray_valid, in_shape, dists, aux_image, pixel_idx, img_n, vid_angle_n, sample_loc_i_n, delta_viewdir_n, frame_weight_n):
        B, R, SR, K, _ = dists.shape
        sampled_Rw2c = sampled_Rw2c.transpose(-1, -2)
        uni_w2c = sampled_Rw2c.dim() == 2
        if not uni_w2c:
            sampled_Rw2c_ray = sampled_Rw2c[:,:,:,0,:,:].view(-1, 3, 3)
            sampled_Rw2c = sampled_Rw2c.reshape(-1, 3, 3)[pnt_mask_flat, :, :]
        pts_ray, pts_pnt = None, None
        if self.opt.agg_feat_xyz_mode != "None" or self.opt.agg_alpha_xyz_mode != "None" or self.opt.agg_color_xyz_mode != "None":
            if self.num_freqs > 0:
                pts = positional_encoding(pts, self.num_freqs)
            pts_ray = pts[ray_valid, :]
            if self.opt.agg_feat_xyz_mode != "None" and self.opt.agg_intrp_order > 0:
                pts_pnt = pts[..., None, :].repeat(1, K, 1).view(-1, pts.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    pts_pnt=pts_pnt[pnt_mask_flat, :]
        viewdirs = viewdirs @ sampled_Rw2c if uni_w2c else (viewdirs[..., None, :] @ sampled_Rw2c_ray).squeeze(-2)
        if self.num_viewdir_freqs > 0:
            viewdirs = positional_encoding(viewdirs, self.num_viewdir_freqs, ori=True)
            ori_viewdirs, viewdirs = viewdirs[..., :3], viewdirs[..., 3:]

        viewdirs = viewdirs[ray_valid, :]

        if self.opt.agg_intrp_order == 0:
            feat = torch.sum(sampled_embedding * weight[..., None], dim=-2)
            feat = feat.view([-1, feat.shape[-1]])[ray_valid, :]
            if self.opt.num_feat_freqs > 0:
                feat = torch.cat([feat, positional_encoding(feat, self.opt.num_feat_freqs)], dim=-1)
            pts = pts_ray
        else:
            dists_flat = dists.view(-1, dists.shape[-1])
            if self.opt.apply_pnt_mask > 0:
                dists_flat = dists_flat[pnt_mask_flat, :]
            dists_flat /= (
                1.0 if self.opt.dist_xyz_deno == 0. else float(self.opt.dist_xyz_deno * np.linalg.norm(vsize)))
            dists_flat[..., :3] = dists_flat[..., :3] @ sampled_Rw2c if uni_w2c else (dists_flat[..., None, :3] @ sampled_Rw2c).squeeze(-2)
            if self.opt.dist_xyz_freq != 0:
                # print(dists.dtype, (self.opt.dist_xyz_deno * np.linalg.norm(vsize)).dtype, dists_flat.dtype)
                dists_flat = positional_encoding(dists_flat, self.opt.dist_xyz_freq)
            feat = sampled_embedding.view(-1, sampled_embedding.shape[-1])
            # print("feat", feat.shape)

            if self.opt.apply_pnt_mask > 0:
                feat = feat[pnt_mask_flat, :]

            if self.opt.num_feat_freqs > 0:
                feat = torch.cat([feat, positional_encoding(feat, self.opt.num_feat_freqs)], dim=-1)
            feat = torch.cat([feat, dists_flat], dim=-1)
            weight = weight.view(B * R * SR, K, 1)
            pts = pts_pnt

        # used_point_embedding = feat[..., : self.opt.point_features_dim]

        if self.opt.agg_feat_xyz_mode != "None":
            feat = torch.cat([feat, pts], dim=-1)
        # print("feat",feat.shape) # 501
        feat = self.block1(feat)

        if self.opt.shading_feature_mlp_layer2 > 0:
            if self.opt.agg_feat_xyz_mode != "None":
                feat = torch.cat([feat, pts], dim=-1)
            if self.opt.agg_intrp_order > 0:
                feat = torch.cat([feat, dists_flat], dim=-1)
            feat = self.block2(feat)

        if self.opt.shading_feature_mlp_layer3 > 0:
            if sampled_color is not None:
                sampled_color = sampled_color.view(-1, sampled_color.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    sampled_color = sampled_color[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_color], dim=-1)
            if sampled_dir is not None:
                sampled_dir = sampled_dir.view(-1, sampled_dir.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    sampled_dir = sampled_dir[pnt_mask_flat, :]
                    sampled_dir = sampled_dir @ sampled_Rw2c if uni_w2c else (sampled_dir[..., None, :] @ sampled_Rw2c).squeeze(-2)
                ori_viewdirs = ori_viewdirs[..., None, :].repeat(1, K, 1).view(-1, ori_viewdirs.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    ori_viewdirs = ori_viewdirs[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_dir - ori_viewdirs, torch.sum(sampled_dir*ori_viewdirs, dim=-1, keepdim=True)], dim=-1)
            feat = self.block3(feat)

        if self.opt.agg_intrp_order == 1:
            import pdb; pdb.set_trace()
            if self.opt.apply_pnt_mask > 0:
                feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
                feat_holder[pnt_mask_flat, :] = feat
            else:
                feat_holder = feat
            feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])
            feat = torch.sum(feat * weight, dim=-2).view([-1, feat.shape[-1]])[ray_valid, :]

            alpha_in = feat
            if self.opt.agg_alpha_xyz_mode != "None":
                alpha_in = torch.cat([alpha_in, pts], dim=-1)

            alpha = self.raw2out_density(self.alpha_branch(alpha_in))

            color_in = feat
            if self.opt.agg_color_xyz_mode != "None":
                color_in = torch.cat([color_in, pts], dim=-1)

            color_in = torch.cat([color_in, viewdirs], dim=-1)
            color_output = self.raw2out_color(self.color_branch(color_in))

            # print("color_output", torch.sum(color_output), color_output.grad)

            output = torch.cat([alpha, color_output], dim=-1)

        elif self.opt.agg_intrp_order == 2:
            alpha_in = feat
            if self.opt.agg_alpha_xyz_mode != "None":
                alpha_in = torch.cat([alpha_in, pts], dim=-1)
            alpha = self.raw2out_density(self.alpha_branch(alpha_in))
            # print(alpha_in.shape, alpha_in)

            if self.opt.apply_pnt_mask > 0:
                alpha_holder = torch.zeros([B * R * SR * K, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
                alpha_holder[pnt_mask_flat, :] = alpha
            else:
                alpha_holder = alpha
            alpha = alpha_holder.view(B * R * SR, K, alpha_holder.shape[-1])
            alpha = torch.sum(alpha * weight, dim=-2).view([-1, alpha.shape[-1]])[ray_valid, :]  # alpha:

            # print("alpha", alpha.shape)
            # alpha_placeholder = torch.zeros([total_len, 1], dtype=torch.float32, device=alpha.device)
            # alpha_placeholder[ray_valid] = alpha

            if self.opt.apply_pnt_mask > 0:
                feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
                feat_holder[pnt_mask_flat, :] = feat
            else:
                feat_holder = feat
            feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])
            feat = torch.sum(feat * weight, dim=-2).view([-1, feat.shape[-1]])[ray_valid, :]

            color_in = feat
            if self.opt.agg_color_xyz_mode != "None":
                color_in = torch.cat([color_in, pts], dim=-1)

            # neural 3D feat branch
            if self.opt.disable_viewdirs:
                color_in = torch.cat([color_in, viewdirs * 0], dim=-1)
            else:
                color_in = torch.cat([color_in, viewdirs], dim=-1)
            color_feature_output = self.color_feature_branch(color_in)  # extract neural 3D feat

            if self.opt.disable_color_feature:
                color_feature_output = 0*color_feature_output

            # image-based feat branch
            if self.opt.dynamic_nearest:
                self.opt.use_nearest = img_n.shape[1]

            if self.opt.use_nearest > 0:
                aux_image = img_n[0].permute(0, 3, 1, 2)
            else:
                aux_image = img_n[0].permute(0, 3, 1, 2)  # the img_n is set to 0 when loading data
            B1, C1, H1, W1 = aux_image.shape

            if self.opt.add_idx:  # frame idx as an additional feat
                aux_vid_vector = torch.zeros((B1, 2, H1, W1), dtype=torch.float32, device=aux_image.device)
                aux_vid_angle = vid_angle_n
                # if train a general network, set the aux_vid_angle to a static number.
                # aux_vid_angle = 0
                aux_vid_vector[:, 0, :, :] = torch.sin(aux_vid_angle[0])[..., None, None]
                aux_vid_vector[:, 1, :, :] = torch.cos(aux_vid_angle[0])[..., None, None]
                aux_feature_s1 = self.aux_block_s1(torch.cat([aux_image, aux_vid_vector], dim=1))
            else:
                aux_feature_s1 = self.aux_block_s1(aux_image)
            aux_feature_s2 = self.aux_block_s2(aux_feature_s1)
            aux_feature_s3 = self.aux_block_s3(aux_feature_s2)
            aux_feature_output = torch.cat([aux_image,
                                           F.interpolate(aux_feature_s1, size=[H1, W1], mode='bilinear'),
                                           F.interpolate(aux_feature_s2, size=[H1, W1], mode='bilinear'),
                                           F.interpolate(aux_feature_s3, size=[H1, W1], mode='bilinear')], dim=1)
            aux_feature_B, aux_feature_C, aux_feature_H, aux_feature_W = aux_feature_output.shape

            drop_positions = None
            ray_drop_positions = None
            if self.opt.use_nearest > 0:
                # version 1, each sampled point on each ray has different aux features according to re-projection.
                sample_loc_i_n = sample_loc_i_n.view(self.opt.use_nearest, -1, 2)
                sample_loc_i_n = sample_loc_i_n[:, ray_valid, :]
                sample_loc_i_n = sample_loc_i_n.view(-1, 2)
                px = sample_loc_i_n[:, 0].to(torch.int32)  # choose pixel position closest to the re-project position. Note, bilinear interpolation will be more accurate.
                py = sample_loc_i_n[:, 1].to(torch.int32)

                delta_viewdir_n = delta_viewdir_n.view(self.opt.use_nearest, -1, 3)
                delta_viewdir_n = delta_viewdir_n[:, ray_valid, :]

                # assign the invalid position as 0
                invalid_mask = torch.ones_like(px).to(torch.float32)
                invalid_xy = torch.where((px < 0) | (px >= W1) | (py < 0) | (py >= H1))
                px[invalid_xy[0]] = 0
                py[invalid_xy[0]] = 0
                invalid_mask[invalid_xy[0]] = 0
                aux_feature_output[:, :, 0, 0] = aux_feature_output[:, :, 0, 0] * 0.0  # (0, 0) as invalid aux feature

                aux_feature_B, aux_feature_C, aux_feature_H, aux_feature_W = aux_feature_output.shape
                px = px.cpu().numpy().astype(np.int32)
                py = py.cpu().numpy().astype(np.int32)
                px = px.reshape(self.opt.use_nearest, -1)
                py = py.reshape(self.opt.use_nearest, -1)
                invalid_mask = invalid_mask.reshape(self.opt.use_nearest, -1)

                # if random_position is 0, use random drop before multi-frame aggregation.
                if self.opt.is_train and self.opt.drop_ratio > 0 and self.opt.random_position == 0:
                    import pdb; pdb.set_trace()
                    if self.opt.ray_points:
                        # ray-based
                        drop_ray_flag = np.zeros((in_shape[1], in_shape[2]))
                        if self.opt.patch_drop:
                            ray_drop_positions = drop_patch_rays(patch_size=int(self.opt.dilation_setup.split('_')[1]),
                                                                 patch_num=int(self.opt.dilation_setup.split('_')[0]),
                                                                 drop_ratio=self.opt.drop_ratio)
                        else:
                            ray_drop_positions = random.sample(range(0, in_shape[1]), int(in_shape[1] * self.opt.drop_ratio))
                        drop_ray_flag[ray_drop_positions, :] = 1
                        drop_ray_flag = drop_ray_flag.flatten()[ray_valid.cpu()]
                        drop_positions = np.where(drop_ray_flag == 1)[0]
                        if self.opt.drop_disturb_range == 0:
                            # the image feature will be set to zero
                            px[:, drop_positions] = 0
                            py[:, drop_positions] = 0
                        else:
                            # drop the image feature by assigning random feats
                            random_sample_px = np.random.randint(0, W1, size=len(drop_positions))
                            random_sample_py = np.random.randint(0, H1, size=len(drop_positions))
                            px[:, drop_positions] = random_sample_px
                            py[:, drop_positions] = random_sample_py
                    else:
                        # query-point based
                        drop_positions = random.sample(range(0, len(px[0])), int(len(px[0]) * self.opt.drop_ratio))
                        if self.opt.drop_disturb_range == 0:
                            # the image feature will be zero
                            px[:, drop_positions] = 0
                            py[:, drop_positions] = 0
                        else:
                            # drop the image feature by assigning random feats
                            random_sample_px = np.random.randint(0, W1, size=(self.opt.use_nearest, len(drop_positions)))
                            random_sample_py = np.random.randint(0, H1, size=(self.opt.use_nearest, len(drop_positions)))
                            px[:, drop_positions] = random_sample_px
                            py[:, drop_positions] = random_sample_py

                # use the 3D feature to guide the multi-frame feat aggregation
                if self.opt.feature_guidance:
                    aux_feature_weight_sum = 0
                    aux_feature_output_sum = 0
                    aux_feature_output_all_tmp = []

                    if self.opt.search_size > 0 and self.opt.search_dilation > 0:
                        # extract image feats from N positions in the image plane
                        raise NotImplementedError
                        # search_size = self.opt.search_size
                        # search_dilation = self.opt.search_dilation
                        # x_offsets, y_offsets = np.meshgrid(np.linspace(-1, 1, search_size), np.linspace(-1, 1, search_size))
                        # x_offsets = np.int32(x_offsets * search_dilation)
                        # y_offsets = np.int32(y_offsets * search_dilation)
                        # for nearest_idx in range(self.opt.use_nearest):
                        #     px_tmp = px[nearest_idx, :]
                        #     py_tmp = py[nearest_idx, :]
                        #     delta_viewdir_n_tmp = delta_viewdir_n[nearest_idx, :, :]
                        #     invalid_mask_tmp = invalid_mask[nearest_idx, :]
                        #     for x_idx in range(search_size):
                        #         for y_idx in range(search_size):
                        #             x_offset = x_offsets[x_idx, y_idx]
                        #             y_offset = y_offsets[x_idx, y_idx]
                        #             px_search_tmp = np.minimum(np.maximum(px_tmp + x_offset, 0), W1-1)
                        #             py_search_tmp = np.minimum(np.maximum(py_tmp + y_offset, 0), H1-1)
                        #             aux_feature_output_tmp = aux_feature_output[nearest_idx:(nearest_idx + 1), :, py_search_tmp, px_search_tmp].permute(0, 2, 1).view([-1, aux_feature_C])
                        #             if self.opt.tradition_attention:
                        #                 raise NotImplementedError
                        #             else:
                        #                 # refine the image feature before aggregation
                        #                 if not self.opt.refine_blend:
                        #                     aux_feature_refined_tmp = aux_feature_output_tmp
                        #                     if self.opt.use_delta_view:
                        #                         aux_feature_weight_tmp = self.aux_merge_weight_block(torch.cat((aux_feature_output_tmp, color_feature_output, delta_viewdir_n_tmp), dim=-1)) * invalid_mask_tmp[..., None]
                        #                     else:
                        #                         aux_feature_weight_tmp = self.aux_merge_weight_block(torch.cat((aux_feature_output_tmp, color_feature_output), dim=-1)) * invalid_mask_tmp[..., None]
                        #                 else:
                        #                     raise NotImplementedError
                        #                     # aux_feature_refined_tmp = self.aux_feature_refine_block(torch.cat((aux_feature_output_tmp, color_feature_output), dim=-1))
                        #                     # aux_feature_weight_tmp = self.aux_merge_weight_block(aux_feature_refined_tmp) * invalid_mask_tmp[..., None]
                        #                 if self.opt.exp_aggregation > 0:
                        #                     aux_feature_weight_tmp = torch.exp(aux_feature_weight_tmp/self.opt.exp_aggregation)
                        #                 aux_feature_output_sum = aux_feature_output_sum + aux_feature_refined_tmp * aux_feature_weight_tmp
                        #                 aux_feature_weight_sum = aux_feature_weight_sum + aux_feature_weight_tmp
                        # if self.opt.tradition_attention:
                        #     # use the traditional attention mechanism
                        #     raise NotImplementedError
                        # else:
                        #     aux_feature_merged = aux_feature_output_sum / (aux_feature_weight_sum + 1e-6)
                    else:
                        # sample image features on the image plane via re-projection
                        for nearest_idx in range(self.opt.use_nearest):
                            px_tmp = px[nearest_idx, :]
                            py_tmp = py[nearest_idx, :]
                            delta_viewdir_n_tmp = delta_viewdir_n[nearest_idx, :, :]
                            invalid_mask_tmp = invalid_mask[nearest_idx, :]
                            aux_feature_output_tmp = aux_feature_output[nearest_idx:(nearest_idx + 1), :, py_tmp, px_tmp].permute(0, 2, 1).view([-1, aux_feature_C])
                            if self.opt.tradition_attention:
                                aux_feature_output_all_tmp.append(aux_feature_output_tmp * invalid_mask_tmp[..., None])
                            else:
                                if not self.opt.refine_blend:
                                    if self.opt.use_delta_view:
                                        aux_feature_weight_tmp = self.aux_merge_weight_block(torch.cat((aux_feature_output_tmp, color_feature_output, delta_viewdir_n_tmp), dim=-1)) * invalid_mask_tmp[..., None]
                                    else:
                                        aux_feature_weight_tmp = self.aux_merge_weight_block(torch.cat((aux_feature_output_tmp, color_feature_output), dim=-1)) * invalid_mask_tmp[..., None]
                                    if self.opt.downweight_blurry_feats:
                                        aux_feature_weight_tmp = aux_feature_weight_tmp * frame_weight_n[0, nearest_idx]
                                    aux_feature_output_sum = aux_feature_output_sum + aux_feature_output_tmp * aux_feature_weight_tmp
                                else:
                                    raise NotImplementedError
                                    # aux_feature_refined_tmp = self.aux_feature_refine_block(torch.cat((aux_feature_output_tmp, color_feature_output), dim=-1))
                                    # aux_feature_weight_tmp = self.aux_merge_weight_block(aux_feature_refined_tmp) * invalid_mask_tmp[..., None]
                                    # if self.opt.downweight_blurry_feats:
                                    #     aux_feature_weight_tmp = aux_feature_weight_tmp * frame_weight_n[0, nearest_idx]
                                    # aux_feature_output_sum = aux_feature_output_sum + aux_feature_refined_tmp * aux_feature_weight_tmp
                                aux_feature_weight_sum = aux_feature_weight_sum + aux_feature_weight_tmp
                        if self.opt.tradition_attention:
                            aux_feature_output_all_tmp = torch.stack(aux_feature_output_all_tmp).permute(1, 2, 0)
                            aux_feature_merged = self.attention_block(color_feature_output[..., None], aux_feature_output_all_tmp, self.opt.use_gumbel_softmax, self.opt.is_train, self.opt.frame_level_attention, self.opt.use_nearest)
                        else:
                            aux_feature_merged = aux_feature_output_sum / (aux_feature_weight_sum + 1e-6)
                else:
                    raise NotImplementedError

                # if random_position is 1; apply random drop after multi-frame aggregation
                if self.opt.is_train and self.opt.drop_ratio > 0 and self.opt.random_position == 1:
                    if self.opt.ray_points:
                        # ray-based
                        drop_ray_flag = np.zeros((in_shape[1], in_shape[2]))
                        if self.opt.drop_patch:
                            ray_drop_positions = drop_patch_rays(patch_size=int(self.opt.dilation_setup.split('_')[1]),
                                                                 patch_num=int(self.opt.dilation_setup.split('_')[0]),
                                                                 drop_ratio=self.opt.drop_ratio)
                        else:
                            ray_drop_positions = random.sample(range(0, in_shape[1]), int(in_shape[1] * self.opt.drop_ratio))
                        drop_ray_flag[ray_drop_positions, :] = 1
                        drop_ray_flag = drop_ray_flag.flatten()[ray_valid.cpu()]
                        drop_positions = np.where(drop_ray_flag == 1)[0]
                        if self.opt.drop_disturb_range == 0:
                            # dropped image feat will be 0
                            aux_feature_merged[drop_positions, :] = aux_feature_merged[drop_positions, :] * 0
                        else:
                            # realize random drop by assigning a random feat
                            random_sample_pb = np.random.randint(0, B1, size=len(drop_positions))
                            random_sample_px = np.random.randint(0, W1, size=len(drop_positions))
                            random_sample_py = np.random.randint(0, H1, size=len(drop_positions))
                            aux_random_feature = aux_feature_output[random_sample_pb, :, random_sample_py, random_sample_px]
                            aux_feature_merged[drop_positions, :] = aux_random_feature + 0
                    else:
                        # query-point-based
                        drop_positions = random.sample(range(0, aux_feature_merged.shape[0]), int(aux_feature_merged.shape[0] * self.opt.drop_ratio))
                        if self.opt.drop_disturb_range == 0:
                            aux_feature_merged[drop_positions, :] = aux_feature_merged[drop_positions, :] * 0
                        else:
                            random_sample_pb = np.random.randint(0, B1, size=len(drop_positions))
                            random_sample_px = np.random.randint(0, W1, size=len(drop_positions))
                            random_sample_py = np.random.randint(0, H1, size=len(drop_positions))
                            aux_random_feature = aux_feature_output[random_sample_pb, :, random_sample_py, random_sample_px]
                            aux_feature_merged[drop_positions, :] = aux_random_feature + 0

            else:
                aux_feature_merged = torch.zeros_like(color_feature_output)[:, 0:aux_feature_C]
                # # use flow-based aux frames, aux img is pre-aligned with rendered image using optical flow.
                # if len(pixel_idx.shape) == 3:
                #     px = pixel_idx.cpu().numpy()[0, :, 0]
                #     py = pixel_idx.cpu().numpy()[0, :, 1]
                #     aux_feature_output = aux_feature_output[:, :, py.astype(np.int32), px.astype(np.int32)]
                #     aux_feature_B, aux_feature_C, aux_feature_HxW = aux_feature_output.shape
                #     aux_feature_output = aux_feature_output.permute(0, 2, 1).view([-1, aux_feature_C])
                #     aux_feature_output = aux_feature_output.unsqueeze(1).repeat(1, SR, 1).view([-1, aux_feature_C])
                # else:
                #     px_py = pixel_idx[0, :, :, 0:2].cpu().numpy()
                #     px = px_py[:, :, 0]
                #     py = px_py[:, :, 1]
                #     aux_feature_output = aux_feature_output[:, :, py.astype(np.int32), px.astype(np.int32)]
                #     aux_feature_B, aux_feature_C, aux_feature_H, aux_feature_W = aux_feature_output.shape
                #     aux_feature_output = aux_feature_output.permute(0, 2, 3, 1).view([-1, aux_feature_C])
                #     aux_feature_output = aux_feature_output.unsqueeze(1).repeat(1, SR, 1).view([-1, aux_feature_C])
                #
                # if len(ray_valid) == aux_feature_output.shape[0]:
                #     aux_feature_output = aux_feature_output[ray_valid, :]
                # else:
                #     aux_feature_output = 0*aux_feature_output[0:len(ray_valid), :]
                #     aux_feature_output = aux_feature_output[ray_valid, :]
                # aux_feature_merged = aux_feature_output

            # mix up features from two branches to generate hybrid feature
            # 'partial' means only part of the color/point feature is used for blending.
            if self.opt.mixup_mode == 'partial':
                color_feature_output_intrinsic = color_feature_output[:, 0:aux_feature_C]
                color_feature_output_view = color_feature_output[:, aux_feature_C:]
                if self.opt.dynamic_weight:
                    blend_weight = self.color_mixup_block(torch.cat((color_feature_output_intrinsic, aux_feature_merged), dim=-1))
                    color_feature_output_intrinsic_mixup = (1-blend_weight) * color_feature_output_intrinsic + blend_weight * aux_feature_merged
                else:
                    color_feature_output_intrinsic_mixup = self.color_mixup_block(torch.cat((color_feature_output_intrinsic, aux_feature_merged), dim=-1))
                if self.opt.learn_residuals:
                    color_feature_output_intrinsic_mixup = color_feature_output_intrinsic_mixup + color_feature_output_intrinsic
                color_feature_output_mixup = torch.cat([color_feature_output_intrinsic_mixup, color_feature_output_view], dim=-1)
            else:
                if self.opt.dynamic_weight:
                    blend_weight = self.color_mixup_block(torch.cat((color_feature_output, aux_feature_merged), dim=-1))
                    color_feature_output_mixup = (1-blend_weight)*color_feature_output + blend_weight*aux_feature_merged
                else:
                    color_feature_output_mixup = self.color_mixup_block(torch.cat((color_feature_output, aux_feature_merged), dim=-1))
                if self.opt.learn_residuals:
                    color_feature_output_mixup = color_feature_output_mixup + color_feature_output

            if self.opt.separate_color_decoder and self.opt.is_train:
                if self.opt.ray_points:
                    drop_ray_ratio = self.opt.drop_ratio
                    drop_ray_flag = np.zeros((in_shape[1], in_shape[2]))
                    if ray_drop_positions is None:
                        ray_drop_positions = random.sample(range(0, in_shape[1]), int(in_shape[1]*drop_ray_ratio))
                    drop_ray_flag[ray_drop_positions, :] = 1
                    drop_ray_flag = drop_ray_flag.flatten()[ray_valid.cpu()]
                    mixup_feature_positions = np.where(drop_ray_flag == 0)[0]
                    point_feature_positions = np.where(drop_ray_flag == 1)[0]
                    mixup_color_output = self.raw2out_color(self.color_final_block(color_feature_output_mixup[mixup_feature_positions, :]))
                    point_color_output = self.raw2out_color(self.color_final_block_2(color_feature_output[point_feature_positions, :]))
                    color_output = torch.zeros([drop_ray_flag.shape[0], self.opt.shading_color_channel_num], dtype=torch.float32, device=color_feature_output_mixup.device)
                    color_output[mixup_feature_positions, :] = mixup_color_output
                    color_output[point_feature_positions, :] = point_color_output
                else:
                    point_ratio = self.opt.drop_ratio
                    drop_flag = np.zeros(color_feature_output_mixup.shape[0])
                    if drop_positions is None:
                        drop_positions = random.sample(range(0, color_feature_output_mixup.shape[0]), int(color_feature_output_mixup.shape[0] * point_ratio))
                    drop_flag[drop_positions] = 1
                    mixup_feature_positions = np.where(drop_flag == 0)[0]
                    point_feature_positions = np.where(drop_flag == 1)[0]
                    mixup_color_output = self.raw2out_color(self.color_final_block(color_feature_output_mixup[mixup_feature_positions, :]))
                    point_color_output = self.raw2out_color(self.color_final_block_2(color_feature_output[point_feature_positions, :]))
                    color_output = torch.zeros([drop_flag.shape[0], self.opt.shading_color_channel_num], dtype=torch.float32, device=color_feature_output_mixup.device)
                    color_output[mixup_feature_positions, :] = mixup_color_output
                    color_output[point_feature_positions, :] = point_color_output
            else:
                color_output = self.raw2out_color(self.color_final_block(color_feature_output_mixup))
            output = torch.cat([alpha, color_output], dim=-1)

        output_placeholder = torch.zeros([total_len, self.opt.shading_color_channel_num*(self.shading_patch_size**2) + 1], dtype=torch.float32, device=output.device)
        output_placeholder[ray_valid] = output
        if self.opt.learnable_blur_kernel_conv:
            blur_kernel_predictor = [self.learn_blur_kernel_conv_block, self.learn_blur_kernel_block]
        else:
            blur_kernel_predictor = self.learn_blur_kernel_block

        return output_placeholder, blur_kernel_predictor

    def print_point(self, dists, sample_loc_w, sampled_xyz, sample_loc, sampled_xyz_pers, sample_pnt_mask):

        # for i in range(dists.shape[0]):
        #     filepath = "./dists.txt"
        #     filepath1 = "./dists10.txt"
        #     filepath2 = "./dists20.txt"
        #     filepath3 = "./dists30.txt"
        #     filepath4 = "./dists40.txt"
        #     dists_cpu = dists.detach().cpu().numpy()
        #     np.savetxt(filepath1, dists_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath2, dists_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath3, dists_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath4, dists_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
        #     dists_cpu = dists[i,...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].detach().cpu().numpy()
        #     np.savetxt(filepath, dists_cpu.reshape(-1, 3), delimiter=";")

        for i in range(sample_loc_w.shape[0]):
            filepath = "./sample_loc_w.txt"
            filepath1 = "./sample_loc_w10.txt"
            filepath2 = "./sample_loc_w20.txt"
            filepath3 = "./sample_loc_w30.txt"
            filepath4 = "./sample_loc_w40.txt"
            sample_loc_w_cpu = sample_loc_w.detach().cpu().numpy()
            np.savetxt(filepath1, sample_loc_w_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sample_loc_w_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sample_loc_w_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sample_loc_w_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            sample_loc_w_cpu = sample_loc_w[i,...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].detach().cpu().numpy()
            np.savetxt(filepath, sample_loc_w_cpu.reshape(-1, 3), delimiter=";")


        for i in range(sampled_xyz.shape[0]):
            sampled_xyz_cpu = sampled_xyz.detach().cpu().numpy()
            filepath = "./sampled_xyz.txt"
            filepath1 = "./sampled_xyz10.txt"
            filepath2 = "./sampled_xyz20.txt"
            filepath3 = "./sampled_xyz30.txt"
            filepath4 = "./sampled_xyz40.txt"
            np.savetxt(filepath1, sampled_xyz_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sampled_xyz_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sampled_xyz_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sampled_xyz_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath, sampled_xyz_cpu[i, ...].reshape(-1, 3), delimiter=";")

        for i in range(sample_loc.shape[0]):
            filepath1 = "./sample_loc10.txt"
            filepath2 = "./sample_loc20.txt"
            filepath3 = "./sample_loc30.txt"
            filepath4 = "./sample_loc40.txt"
            filepath = "./sample_loc.txt"
            sample_loc_cpu =sample_loc.detach().cpu().numpy()

            np.savetxt(filepath1, sample_loc_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sample_loc_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sample_loc_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sample_loc_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath, sample_loc[i, ...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].reshape(-1, 3).detach().cpu().numpy(), delimiter=";")

        for i in range(sampled_xyz_pers.shape[0]):
            filepath1 = "./sampled_xyz_pers10.txt"
            filepath2 = "./sampled_xyz_pers20.txt"
            filepath3 = "./sampled_xyz_pers30.txt"
            filepath4 = "./sampled_xyz_pers40.txt"
            filepath = "./sampled_xyz_pers.txt"
            sampled_xyz_pers_cpu = sampled_xyz_pers.detach().cpu().numpy()

            np.savetxt(filepath1, sampled_xyz_pers_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sampled_xyz_pers_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sampled_xyz_pers_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sampled_xyz_pers_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")

            np.savetxt(filepath, sampled_xyz_pers_cpu[i, ...].reshape(-1, 3), delimiter=";")
        print("saved sampled points and shading points")
        exit()


    def gradiant_clamp(self, sampled_conf, min=0.0001, max=1):
        diff = sampled_conf - torch.clamp(sampled_conf, min=min, max=max)
        return sampled_conf - diff.detach()


    def forward(self, sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, vsize, grid_vox_sz, aux_image=None, pixel_idx=None, img_n=None, vid_angle_n=None, sample_loc_i_n=None, delta_viewdir_n=None, frame_weight_n=None):
        # return B * R * SR * channel
        '''
        :param sampled_conf: B x valid R x SR x K x 1
        :param sampled_embedding: B x valid R x SR x K x F
        :param sampled_xyz_pers:  B x valid R x SR x K x 3
        :param sampled_xyz:       B x valid R x SR x K x 3
        :param sample_pnt_mask:   B x valid R x SR x K
        :param sample_loc:        B x valid R x SR x 3
        :param sample_loc_w:      B x valid R x SR x 3
        :param sample_ray_dirs:   B x valid R x SR x 3
        :param vsize:
        :return:
        '''
        ray_valid = torch.any(sample_pnt_mask, dim=-1).view(-1)
        total_len = len(ray_valid)
        in_shape = sample_loc_w.shape
        if total_len == 0 or torch.sum(ray_valid) == 0:
            # print("skip since no valid ray, total_len:", total_len, torch.sum(ray_valid))
            return torch.zeros(in_shape[:-1] + (self.opt.shading_color_channel_num + 1,), device=ray_valid.device, dtype=torch.float32), ray_valid.view(in_shape[:-1]), None, None
        if self.opt.agg_dist_pers < 0:
            dists = sample_loc_w[..., None, :]
        elif self.opt.agg_dist_pers == 0:
            dists = sampled_xyz - sample_loc_w[..., None, :]
        elif self.opt.agg_dist_pers == 1:
            dists = sampled_xyz_pers - sample_loc[..., None, :]
        elif self.opt.agg_dist_pers == 2:
            if sampled_xyz_pers.shape[1] > 0:
                xdist = sampled_xyz_pers[..., 0] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]
                ydist = sampled_xyz_pers[..., 1] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]
                zdist = sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 2]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 3], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.opt.agg_dist_pers == 10:

            if sampled_xyz_pers.shape[1] > 0:
                dists = sampled_xyz_pers - sample_loc[..., None, :]
                dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 6], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.opt.agg_dist_pers == 20:

            if sampled_xyz_pers.shape[1] > 0:
                xdist = sampled_xyz_pers[..., 0] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]
                ydist = sampled_xyz_pers[..., 1] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]
                zdist = sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 2]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)
                # dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
                dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 6], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.opt.agg_dist_pers == 30:

            if sampled_xyz_pers.shape[1] > 0:
                w_dists = sampled_xyz - sample_loc_w[..., None, :]
                dists = torch.cat([torch.sum(w_dists*sample_ray_dirs[..., None, :], dim=-1, keepdim=True), dists], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 4], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)
        else:
            print("illegal agg_dist_pers code: ", agg_dist_pers)
            exit()
        # self.print_point(dists, sample_loc_w, sampled_xyz, sample_loc, sampled_xyz_pers, sample_pnt_mask)

        weight, sampled_embedding = self.dist_func(sampled_embedding, dists, sample_pnt_mask, vsize, grid_vox_sz, axis_weight=self.axis_weight)

        if self.opt.agg_weight_norm > 0 and self.opt.agg_distance_kernel != "trilinear" and not self.opt.agg_distance_kernel.startswith("num"):
            weight = weight / torch.clamp(torch.sum(weight, dim=-1, keepdim=True), min=1e-8)

        pnt_mask_flat = sample_pnt_mask.view(-1)
        pts = sample_loc_w.view(-1, sample_loc_w.shape[-1])
        viewdirs = sample_ray_dirs.view(-1, sample_ray_dirs.shape[-1])
        conf_coefficient = 1
        if sampled_conf is not None:
            conf_coefficient = self.gradiant_clamp(sampled_conf[..., 0], min=0.0001, max=1)
        output, blur_predictor = getattr(self, self.which_agg_model, None)(sampled_color, sampled_Rw2c, sampled_dir, sampled_conf,
                                                                           sampled_embedding, sampled_xyz_pers, sampled_xyz,
                                                                           sample_pnt_mask, sample_loc, sample_loc_w,
                                                                           sample_ray_dirs, vsize, weight * conf_coefficient,
                                                                           pnt_mask_flat, pts, viewdirs, total_len, ray_valid,
                                                                           in_shape, dists, aux_image, pixel_idx, img_n, vid_angle_n,
                                                                           sample_loc_i_n, delta_viewdir_n, frame_weight_n)
        if (self.opt.sparse_loss_weight <= 0) and ("conf_coefficient" not in self.opt.zero_one_loss_items) and self.opt.prob == 0:
            weight, conf_coefficient = None, None

        if self.opt.is_train:
            return output.view(in_shape[:-1] + (self.opt.shading_color_channel_num * (self.shading_patch_size ** 2) + 1,)), ray_valid.view(in_shape[:-1]), weight, conf_coefficient, blur_predictor
        else:
            return output.view(in_shape[:-1] + (self.opt.shading_color_channel_num * (self.shading_patch_size ** 2) + 1,)), ray_valid.view(in_shape[:-1]), weight, conf_coefficient


import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import format as fmt
import os, math, cv2
from .base_model import BaseModel

from .rendering.diff_render_func import find_render_function, find_blend_function, find_tone_map, alpha_blend
from .rendering.diff_ray_marching import find_ray_generation_method, find_refined_ray_generation_method, ray_march, \
    alpha_ray_march
from utils import format as fmt
from utils.spherical import SphericalHarm, SphericalHarm_table
from utils.util import add_property2dict
from torch.autograd import Variable

from pytorch_msssim import ssim, ms_ssim, SSIM

from PIL import Image


def mse2psnr(x): return -10. * torch.log(x) / np.log(10.)


ssim_module = SSIM(win_size=7, win_sigma=1.5, data_range=1, size_average=True, channel=3)

class BaseRenderingModel(BaseModel):
    ''' A base rendering model that provides the basic loss functions,
        selctions of different rendering functions, ray generation functions,
        blending functions (for collocated and non-collocated ray marching),
        and functions to setup encoder and decoders.
        A sub model needs to at least re-implement create_network_models() and run_network_models() for actual rendering.
        Examples are: hirarchical_volumetric_model etc.

        The model collects
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # loss parameters
        parser.add_argument(
            "--sparse_loss_weight",
            type=float,
            default=0,
            help="The (multiple) output items to supervise with gt color.")
        parser.add_argument(
            "--color_loss_items",
            type=str,
            nargs='+',
            default=None,
            help="The (multiple) output items to supervise with gt color.")
        parser.add_argument(
            "--test_color_loss_items",
            type=str,
            nargs='+',
            default=None,
            help="The (multiple) output items to supervise with gt color.")
        parser.add_argument(
            "--color_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each color supervision item. The number of this args should be 1 or match the number in --color_loss_items"
        )
        parser.add_argument(
            "--bg_loss_items",
            type=str,
            nargs='+',
            default=[],
            help="The (multiple) output items to supervise with gt masks.")
        parser.add_argument(
            "--bg_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each mask supervision item. The number of this args should be 1 or match the number in --bg_loss_items"
        )
        parser.add_argument(
            "--depth_loss_items",
            type=str,
            nargs='+',
            default=[],
            help="The (multiple) output items to supervise with gt depth.")
        parser.add_argument(
            "--depth_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each depth supervision item. The number of this args should be 1 or match the number in --depth_loss_items"
        )
        parser.add_argument(
            "--zero_one_loss_items",
            type=str,
            nargs='+',
            default=[],
            help=
            "The (multiple) output items to regularize to be close to either 0 or 1 ."
        )
        parser.add_argument(
            "--zero_one_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each zero_one regularization item. The number of this args should be 1 or match the number in --zero_one_loss_items"
        )

        parser.add_argument(
            "--l2_size_loss_items",
            type=str,
            nargs='+',
            default=[],
            help=
            "The (multiple) output items to regularize to be close to either 0 or 1 ."
        )
        parser.add_argument(
            "--l2_size_loss_weights",
            type=float,
            nargs='+',
            default=[0.0],
            help=
            "The weights for each zero_one regularization item. The number of this args should be 1 or match the number in --zero_one_loss_items"
        )
        parser.add_argument(
            "--zero_epsilon",
            type=float,
            default=1e-3,
            help="epsilon in logarithmic regularization terms when needed.",
        )
        parser.add_argument(
            "--no_loss",
            type=int,
            default=False,
            help="do not compute loss.",
        )

        # visualization terms
        parser.add_argument(
            "--visual_items",
            type=str,
            nargs='*',
            default=None,
            help=
            "The (multiple) output items to show as images. This will replace the default visual items"
        )
        parser.add_argument(
            "--visual_items_additional",
            type=str,
            nargs='+',
            default=[],
            help=
            "The (multiple) output items to show as images in addition to default items. This is ignored if --visual_iterms is used"
        )
        parser.add_argument(
            '--out_channels',
            type=int,
            default=None,
            help=
            'number of output channels in decoder; default 4 for radiance, 8 for microfacet and others'
        )
        # ray generation
        parser.add_argument(
            '--which_ray_generation',
            type=str,
            default='cube',
            help='which ray point generation method to use [cube]')
        parser.add_argument('--domain_size',
                            type=int,
                            default=1,
                            help='Size of the ray marching domain')
        # rendering functions
        parser.add_argument('--which_render_func',
                            type=str,
                            default='microfacet',
                            help='which render method to use')
        parser.add_argument(
            '--which_blend_func',
            type=str,
            default='alpha',
            help=
            'which blend function to use. Hint: alpha2 for collocated, alpha for non-collocated'
        )
        parser.add_argument('--which_tonemap_func',
                            type=str,
                            default='gamma',
                            help='which tone map function to use.')

        parser.add_argument(
            '--num_pos_freqs',
            type=int,
            default=-1,
            help=
            'number of frequency for position encoding if using nerf or mixed mlp decoders'
        )
        parser.add_argument(
            '--num_viewdir_freqs',
            type=int,
            default=-1,
            help=
            'number of frequency for view direction encoding if using nerf decoders'
        )
        parser.add_argument(
            '--num_feature_freqs',
            type=int,
            default=-1,
            help=
            'number of frequency for feature encoding if using mixed mlp decoders'
        )
        parser.add_argument(
            '--boundary_mode',
            type=int,
            default=0,
            help=
            'how to deal with pixels around the patch boundary when applying blur kernels'
        )

        return parser

    def add_default_color_losses(self, opt):
        ''' if no color loss terms are specified, this function is called to
            add default supervision into opt.color_loss_items
        '''

        opt.color_loss_items = []  # add this to actual names in subclasses

    def add_default_visual_items(self, opt):
        ''' if no visual terms are specified, this function is called to
            add default visualization items
        '''
        opt.visual_items = ['gt_image']  # add this to actual names in subclasses

    def check_setup_loss(self, opt):
        ''' this function check and setup all loss items and weights.'''

        self.loss_names = ['total']
        if not opt.color_loss_items:
            self.add_default_color_losses(opt)
        if len(opt.color_loss_weights) != 1 and len(
                opt.color_loss_weights) != len(opt.color_loss_items):
            print(fmt.RED + "color_loss_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.color_loss_weights) == 1 and len(opt.color_loss_items) > 1:
            opt.color_loss_weights = np.ones(len(
                opt.color_loss_items), np.float32) * opt.color_loss_weights[0]
        self.loss_names += opt.color_loss_items

        if len(opt.depth_loss_weights) != 1 and len(
                opt.depth_loss_weights) != len(opt.depth_loss_items):
            print(fmt.RED + "color_depth_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.depth_loss_weights) == 1 and len(opt.depth_loss_items) > 1:
            opt.depth_loss_weights = np.ones(len(
                opt.depth_loss_items), np.float32) * opt.depth_loss_weights[0]
        self.loss_names += opt.depth_loss_items

        if len(opt.zero_one_loss_weights) != len(
                opt.zero_one_loss_items) and len(
            opt.zero_one_loss_weights) != 1:
            print(fmt.RED + "zero_one_loss_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.zero_one_loss_weights) == 1 and len(
                opt.zero_one_loss_items) > 1:
            opt.zero_one_loss_weights = np.ones(
                len(opt.zero_one_loss_items),
                np.float32) * opt.zero_one_loss_weights[0]
        self.loss_names += opt.zero_one_loss_items

        if len(opt.bg_loss_weights) != 1 and len(opt.bg_loss_weights) != len(
                opt.bg_loss_items):
            print(fmt.RED + "bg_loss_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.bg_loss_weights) == 1 and len(opt.bg_loss_items) > 1:
            opt.bg_loss_weights = np.ones(len(opt.bg_loss_items),
                                          np.float32) * opt.bg_loss_weights[0]
        self.loss_names += opt.bg_loss_items
        if opt.sparse_loss_weight > 0:
            self.loss_names += ["sparse"]

        # add the functions used in losses
        self.l1loss = torch.nn.L1Loss().to(self.device)
        self.l2loss = torch.nn.MSELoss().to(self.device)

    def check_setup_visuals(self, opt):
        if opt.visual_items is None:
            print("visual_items not ", opt.visual_items)
            self.add_default_visual_items(opt)
            self.visual_names += opt.visual_items
            self.visual_names += opt.visual_items_additional
        else:
            self.visual_names += opt.visual_items

        if len(self.visual_names) == 0:
            print(fmt.YELLOW + "No items are visualized" + fmt.END)

    def create_network_models(self, opt):
        '''
        This function should create the rendering networks.
        Every subnetwork model needs to be named as self.net_"name",
        and the "name" needs to be added to the self.model_names list.
        An example of this is like:
            self.model_names = ['ray_marching']
            self.net_ray_marching = network_torch_model(self.opt)

            if self.opt.gpu_ids:
                self.net_ray_marching.to(self.device)
                self.net_ray_marching = torch.nn.DataParallel(
                    self.net_ray_marching, self.opt.gpu_ids)
        '''
        pass

    def run_network_models(self):
        '''
        This function defines how the network is run.
        This function should use the self.input as input to the network.
        and return a dict of output (that will be assign to self.output).
        If only a sinlge network is used, this function could be simply just:
            return net_module(**self.input)
        '''
        raise NotImplementedError()

    def prepare_network_parameters(self, opt):
        '''
        Setup the parameters the network is needed.
        By default, it finds rendering (shading) function, ray generation function, tonemap function, etc.
        '''

        self.check_setup_loss(opt)

        if len(self.loss_names) == 1 and opt.is_train == True:
            print(fmt.RED + "Requiring losses to train" + fmt.END)
            raise NotImplementedError()

        self.check_setup_visuals(opt)

        self.check_setup_renderFunc_channels(opt)

        self.blend_func = find_blend_function(opt.which_blend_func)
        self.raygen_func = find_ray_generation_method(opt.which_ray_generation)
        self.tonemap_func = find_tone_map(opt.which_tonemap_func)

        self.found_funcs = {}
        add_property2dict(
            self.found_funcs, self,
            ["blend_func", "raygen_func", "tonemap_func", "render_func"])

    def setup_optimizer(self, opt):
        '''
            Setup the optimizers for all networks.
            This assumes network modules have been added to self.model_names
            By default, it uses an adam optimizer for all parameters.
        '''

        params = []
        for name in self.model_names:
            net = getattr(self, 'net_' + name)
            params = params + list(net.parameters())

        self.optimizers = []

        self.optimizer = torch.optim.Adam(params,
                                          lr=opt.lr,
                                          betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer)

    def check_opts(self, opt):
        pass

    def initialize(self, opt):
        super(BaseRenderingModel, self).initialize(opt)
        self.opt = opt

        if self.is_train:
            self.check_opts(opt)
        self.prepare_network_parameters(opt)

        self.create_network_models(opt)

        # check model creation
        if not self.model_names:
            print(
                fmt.RED +
                "No network is implemented! Or network's name is not properly added to self.model_names"
                + fmt.END)
            raise NotImplementedError()
        for mn in self.model_names:
            if not hasattr(self, "net_" + mn):
                print(fmt.RED + "Network " + mn + " is missing" + fmt.END)
                raise NotImplementedError()

        # setup optimizer
        if self.is_train:
            self.setup_optimizer(opt)

        self.xv_patches, self.yv_patches = [], []

    def set_input(self, input):

        # setup self.input
        # this dict is supposed to be sent the network via **self.input in run_network_modules
        self.input = input
        for key, item in self.input.items():
            if isinstance(item, torch.Tensor):
                self.input[key] = item.to(self.device)

        # gt required in loss compute
        self.gt_image = self.input['gt_image'].to(
            self.device) if 'gt_image' in input else None

        self.gt_image_patch = self.input['gt_image_patch'].to(
            self.device) if 'gt_image_patch' in input else None

        self.gt_image_full = self.input['gt_image_full'].to(
            self.device) if 'gt_image_full' in input else None

        self.pixel_index = self.input['pixel_idx'].to(
            self.device) if 'pixel_idx' in input else None

        self.gt_depth = self.input['gt_depth'].to(
            self.device) if 'gt_depth' in input else None

        self.gt_mask = self.input['gt_mask'].to(
            self.device) if 'gt_mask' in input else None

        # pre-defined blur-kernels
        self.blur_kernels = self.input['blur_kernels'].to(
            self.device) if 'blur_kernels' in input else None

        # dilation setting
        self.dilation_PatchNum = self.input['dilation_PatchNum'].to(
            self.device)[0] if 'dilation_PatchNum' in input else None

        self.dilation_PatchSize = self.input['dilation_PatchSize'].to(
            self.device)[0] if 'dilation_PatchSize' in input else None

        # frame weight
        self.frame_weight = self.input['frame_weight'].cpu().numpy()[0] if 'frame_weight' in input else None

    def set_visuals(self):
        for key, item in self.output.items():
            if key in self.visual_names:
                setattr(self, key, item)
        if "coarse_raycolor" not in self.visual_names:
            key = "coarse_raycolor"
            setattr(self, key, self.output[key])

    def check_setup_renderFunc_channels(self, opt):
        ''' Find render functions;
            the function is often used by subclasses when creating rendering networks.
        '''

        self.render_func = find_render_function(opt.which_render_func)

        if opt.which_render_func == 'radiance':
            if opt.out_channels is None:
                opt.out_channels = 4
        elif opt.which_render_func == 'microfacet':
            if opt.out_channels is None:
                opt.out_channels = 8
        elif opt.which_render_func == 'harmonics':
            if opt.out_channels is None:
                opt.out_channels = 1 + 3 * 5 * 5
            deg = int(((opt.out_channels - 1) / 3) ** 0.5)
            if 1 + deg * deg * 3 != opt.out_channels:
                print(
                    fmt.RED +
                    '[Error] output channels should match the number of sh basis'
                    + fmt.END)
                exit()
            if deg <= 5:
                print("using SH table")
                self.shcomputer = SphericalHarm_table(deg)
            else:
                print("using runtime SH")
                self.shcomputer = SphericalHarm(deg)
            self.render_func.sphericalHarm = self.shcomputer
        else:
            if opt.out_channels is None:
                opt.out_channels = 8
        self.out_channels = opt.out_channels

    def check_getDecoder(self, opt, **kwargs):
        '''construct a decoder; this is often used by subclasses when creating networks.'''

        decoder = None
        if opt.which_decoder_model == 'mlp':
            decoder = MlpDecoder(num_freqs=opt.num_pos_freqs,
                                 out_channels=opt.out_channels,
                                 **kwargs)
        elif opt.which_decoder_model == 'viewmlp':
            decoder = ViewMlpDecoder(num_freqs=opt.num_pos_freqs,
                                     num_viewdir_freqs=opt.num_viewdir_freqs,
                                     num_channels=opt.out_channels,
                                     **kwargs)
        elif opt.which_decoder_model == 'viewmlpsml':
            decoder = ViewMlpSmlDecoder(num_freqs=opt.num_pos_freqs,
                                        num_viewdir_freqs=opt.num_viewdir_freqs,
                                        num_channels=opt.out_channels,
                                        **kwargs)
        elif opt.which_decoder_model == 'viewmlpmid':
            decoder = ViewMlpMidDecoder(num_freqs=opt.num_pos_freqs,
                                        num_viewdir_freqs=opt.num_viewdir_freqs,
                                        num_channels=opt.out_channels,
                                        **kwargs)
        elif opt.which_decoder_model == 'nv_mlp':
            decoder = VolumeDecoder(256,
                                    template_type=opt.nv_template_type,
                                    template_res=opt.nv_resolution,
                                    out_channels=opt.out_channels,
                                    **kwargs)
        elif opt.which_decoder_model == 'discrete_microfacet':
            decoder = DiscreteVolumeMicrofacetDecoder(
                opt.discrete_volume_folder,
                out_channels=opt.out_channels,
                **kwargs)
        elif opt.which_decoder_model == 'discrete_general':
            decoder = DiscreteVolumeGeneralDecoder(
                opt.discrete_volume_folder,
                out_channels=opt.out_channels,
                **kwargs)
        elif opt.which_decoder_model == 'mixed_mlp':
            decoder = MixedDecoder(256,
                                   template_type=opt.nv_template_type,
                                   template_res=opt.nv_resolution,
                                   mlp_channels=128,
                                   out_channels=opt.out_channels,
                                   position_freqs=opt.num_pos_freqs,
                                   feature_freqs=opt.num_feature_freqs,
                                   **kwargs)
        elif opt.which_decoder_model == 'mixed_separate_code':
            decoder = MixedSeparatedDecoder(
                256,
                template_type=opt.nv_template_type,
                template_res=opt.nv_resolution,
                mlp_channels=128,
                out_channels=opt.out_channels,
                position_freqs=opt.num_pos_freqs,
                feature_freqs=opt.num_feature_freqs,
                **kwargs)
        else:
            raise RuntimeError('Unknown decoder model: ' +
                               opt.which_decoder_model)

        return decoder

    def forward(self):
        import pdb; pdb.set_trace()
        self.output = self.run_network_models()

        self.set_visuals()

        if not self.opt.no_loss:
            if self.blur_kernels is not None:
                self.blur_update_output()
            self.compute_losses()

    def save_image(self, img_array, filepath):
        assert len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] in [3, 4])

        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        Image.fromarray(img_array).save(filepath)

    def patch_based_reshape_forward(self, input_large):
        """
        large patch to small patches
        reshape spatial H, W to B
        :param input_large: 1x3xHxW
        :return: Bx3xH'xW'
        """
        input_patches = []
        for patch_i in range(self.dilation_PatchNum):
            for patch_j in range(self.dilation_PatchNum):
                input_patch = input_large[0, :,
                              patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                              patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                input_patches.append(input_patch)
        input_patches = torch.stack(input_patches)
        return input_patches

    def patch_based_reshape_backward(self, input_patches):
        """
        small patches to large patch
        reshape B to spatial H, W
        :param input_large: Bx3xH'xW'
        :return: 1x3xHxW
        """
        cnt = 0
        sample_size = self.dilation_PatchSize * self.dilation_PatchNum
        input_large = torch.ones((1, 3, sample_size, sample_size), requires_grad=True)
        for patch_i in range(self.dilation_PatchNum):
            for patch_j in range(self.dilation_PatchNum):
                input_large[0, :, patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize] = input_patches[cnt]
                cnt = cnt + 1
        return input_large

    def detail_aware_weights(self, output_rgb_in, gt_rgb_in, blur_k=3, which_blur='gaussian', edge_k=3,
                             which_edge='laplacian', stop_grad=True, scale=50, up=4.0, low=0.25):
        """
        :param output: output, 1xNx3
        :param gt: GT, 1xNx3
        :param blur_k:
        :param which_blur: 'gaussian'
        :param edge_k:
        :param which_edge: 'laplacian'
        :return:
        """
        if stop_grad:
            gt_rgb = gt_rgb_in.clone().detach()
            output_rgb = output_rgb_in.clone().detach()
        else:
            gt_rgb = gt_rgb_in
            output_rgb = output_rgb_in

        sample_size = self.dilation_PatchSize * self.dilation_PatchNum
        gt_rgb = gt_rgb.reshape(1, sample_size, sample_size, -1).permute(0, 3, 1, 2)  # 1x3xHxW
        output_rgb = output_rgb.reshape(1, sample_size, sample_size, -1).permute(0, 3, 1, 2)  # 1x3xHxW
        # output_patches_0 = self.patch_based_reshape_forward(output_rgb)
        # gt_patches_0 = self.patch_based_reshape_forward(gt_rgb)

        # alternatively
        output_patches = output_rgb[0].view(1, 3, self.dilation_PatchNum, self.dilation_PatchSize,
                                            self.dilation_PatchNum, self.dilation_PatchSize) \
            .permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, self.dilation_PatchSize, self.dilation_PatchSize)  # Bx3xH'xW'
        gt_patches = gt_rgb[0].view(1, 3, self.dilation_PatchNum, self.dilation_PatchSize, self.dilation_PatchNum,
                                    self.dilation_PatchSize) \
            .permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, self.dilation_PatchSize, self.dilation_PatchSize)  # Bx3xH'xW'

        # change to gray_scale, average. (more accurate: 0.7xxx + xxx + xxx)
        output_patches_gray = torch.mean(output_patches, dim=1, keepdims=True)
        gt_patches_gray = torch.mean(gt_patches, dim=1, keepdims=True)  # Bx1xH'xW'

        # 3x3 gaussian kernel
        blur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        blur_kernel = torch.from_numpy(blur_kernel)[None, None, ...]
        blur_kernel = nn.Parameter(data=blur_kernel.to(torch.float32), requires_grad=False).cuda()

        # 3x3 laplacian edge
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edge_kernel = torch.from_numpy(edge_kernel)[None, None, ...]
        edge_kernel = nn.Parameter(data=edge_kernel.to(torch.float32), requires_grad=False).cuda()

        # apply conv
        output_patches_gray_blur = F.conv2d(output_patches_gray, blur_kernel, padding=int(3 / 2))
        output_patches_gray_edge = torch.abs(F.conv2d(output_patches_gray_blur, edge_kernel, padding=int(3 / 2)))
        gt_patches_gray_blur = F.conv2d(gt_patches_gray, blur_kernel, padding=int(3 / 2))
        gt_patches_gray_edge = torch.abs(F.conv2d(gt_patches_gray_blur, edge_kernel, padding=int(3 / 2)))
        edge_diff = torch.mean(gt_patches_gray_edge, dim=(1, 2, 3)) - torch.mean(output_patches_gray_edge + 1e-10, dim=(1, 2, 3))

        # eliminate the influence of padding, fake edge.
        """
        remove boundary pixels/edges of a patch.
        """

        # transform to weight map.
        weight_map = torch.zeros_like(output_patches) + edge_diff[..., None, None, None].repeat(1, 3, output_patches.shape[2], output_patches.shape[3])

        # reshape back
        weight_map = weight_map.reshape(1, self.dilation_PatchNum, self.dilation_PatchNum, 3, self.dilation_PatchSize,
                                        self.dilation_PatchSize).permute(0, 3, 1, 4, 2, 5) \
            .reshape(1, 3, sample_size, sample_size).permute(0, 2, 3, 1).reshape(1, -1, 3)

        weight_map = torch.clamp(1 + weight_map * scale, low, up)

        return weight_map

    def blur_update_output(self, faster_version=True):
        if self.dilation_PatchNum > 0:
            sample_size = self.dilation_PatchNum * self.dilation_PatchSize
            gt_rgb_original = self.gt_image.reshape(1, sample_size, sample_size, -1).permute(0, 3, 1, 2)
            output_rgb_original = self.output["coarse_raycolor"].reshape(1, sample_size, sample_size, -1).permute(0, 3, 1, 2)
            gt_rgb_patches = []
            output_rgb_patches = []

            # reshape 1xCxHxW to BxCxH'xW', H' and W' are small patch sizes.
            if faster_version:
                xv_patches = self.xv_patches
                yv_patches = self.yv_patches
                if len(xv_patches) == 0 or len(yv_patches) == 0:
                    x = np.arange(0, int(sample_size))
                    y = np.arange(0, int(sample_size))
                    xv, yv = np.meshgrid(x, y)
                    for patch_i in range(self.dilation_PatchNum):
                        for patch_j in range(self.dilation_PatchNum):
                            xv_patch = xv[patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                                       patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                            yv_patch = yv[patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                                       patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                            xv_patches.append(xv_patch)
                            yv_patches.append(yv_patch)
                    self.xv_patches = np.stack(xv_patches)
                    self.yv_patches = np.stack(yv_patches)
                gt_rgb_patches = gt_rgb_original[0, :, yv_patches, xv_patches].permute(1, 0, 2, 3)
                output_rgb_patches_best = output_rgb_original[0, :, yv_patches, xv_patches].permute(1, 0, 2, 3)
            else:
                raise NotImplementedError
                # import pdb; pdb.set_trace()
                # for patch_i in range(self.dilation_PatchNum):
                #     for patch_j in range(self.dilation_PatchNum):
                #         gt_rgb_patch = gt_rgb_original[0, :,
                #                        patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                #                        patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                #         output_rgb_patch = output_rgb_original[0, :,
                #                            patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                #                            patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                #         gt_rgb_patches.append(gt_rgb_patch)
                #         output_rgb_patches.append(output_rgb_patch)
                # gt_rgb_patches = torch.stack(gt_rgb_patches)
                # output_rgb_patches_best = torch.stack(output_rgb_patches)

            # simulate blur
            if faster_version:
                B, N, H, W = self.blur_kernels.shape
                output_rgb_patches_best = output_rgb_patches_best.reshape(self.dilation_PatchNum*self.dilation_PatchNum*3, 1, self.dilation_PatchSize, self.dilation_PatchSize)
                masks = torch.ones_like(output_rgb_patches_best)
                blur_kernels = self.blur_kernels.permute(1, 0, 2, 3).cuda()
                masks_out = F.conv2d(masks, blur_kernels, padding=int(H / 2))
                output_rgb_patches_best_blurred = F.conv2d(output_rgb_patches_best, blur_kernels, padding=int(H / 2)) / masks_out
                output_rgb_patches_best_blurred = torch.cat((output_rgb_patches_best_blurred, output_rgb_patches_best), dim=1)  # contain the original output patch
                output_rgb_patches_best_blurred = output_rgb_patches_best_blurred.reshape(self.dilation_PatchNum*self.dilation_PatchNum, 3, N+1, self.dilation_PatchSize, self.dilation_PatchSize)
                diff = torch.sum(torch.abs(output_rgb_patches_best_blurred - torch.tile(gt_rgb_patches[None, ...].permute(1, 2, 0, 3, 4), (1, 1, N+1, 1, 1))), dim=(1, 3, 4))
                select_index = torch.argmin(diff, dim=1)
                output_rgb_patches_best = output_rgb_patches_best_blurred[np.arange(int(self.dilation_PatchNum*self.dilation_PatchNum)), :, list(select_index), :, :]

                """
                The above blur simulation ways cannot correctly handle pixels around patch boundary.
                To improve:
                sample a large patch 16x16, split it into 4 8x8 patches.
                utilize rendered pixels from adjacent patches when operating around patch boundary.  
                ...
                """
            else:
                raise NotImplementedError
                # import pdb; pdb.set_trace()
                # B, N, H, W = self.blur_kernels.shape
                # loss_best = torch.sum(torch.abs(output_rgb_patches_best - gt_rgb_patches), dim=(1, 2, 3))
                # c1 = output_rgb_patches_best[:, 0:1, :, :].clone()
                # c2 = output_rgb_patches_best[:, 1:2, :, :].clone()
                # c3 = output_rgb_patches_best[:, 2:3, :, :].clone()
                # mask = torch.ones(c1.shape, device=c1.device)
                # for i in range(N):
                #     blur_kernal = self.blur_kernels[:, i:i + 1, :, :]
                #     blur_kernal = nn.Parameter(data=blur_kernal, requires_grad=False).cuda()
                #     mask_out = F.conv2d(mask, blur_kernal, padding=int(H / 2))  # avoid the influence of zero padding.
                #     c1_out = F.conv2d(c1, blur_kernal, padding=int(H / 2)) / mask_out
                #     c2_out = F.conv2d(c2, blur_kernal, padding=int(H / 2)) / mask_out
                #     c3_out = F.conv2d(c3, blur_kernal, padding=int(H / 2)) / mask_out
                #     blur_output = torch.cat([c1_out, c2_out, c3_out], dim=1)
                #     loss_tmp = torch.sum(torch.abs(blur_output - gt_rgb_patches), dim=(1, 2, 3))
                #
                #     select_index = torch.where(loss_tmp < loss_best)[0]
                #     output_rgb_patches_best[select_index, :, :, :] = blur_output[select_index, :, :, :]
                #     loss_best[select_index] = loss_tmp[select_index]

            # reshape back
            if faster_version:
                output_rgb_new = []
                blurred_rgb_patches_list = list(torch.chunk(output_rgb_patches_best, 1, dim=0)[0])
                for col_j in range(self.dilation_PatchNum):
                    output_rgb_new.append(torch.cat(blurred_rgb_patches_list[col_j*self.dilation_PatchNum:(col_j+1)*self.dilation_PatchNum], dim=-1))
                output_rgb_new = torch.cat(output_rgb_new, dim=-2)[None, ...]
                self.output["coarse_raycolor"] = output_rgb_new.permute(0, 2, 3, 1).view(1, -1, 3)
            else:
                raise NotImplementedError
                # import pdb; pdb.set_trace()
                # cnt = 0
                # output_rgb_new = []
                # for patch_i in range(self.dilation_PatchNum):
                #     output_rgb_col = []
                #     for patch_j in range(self.dilation_PatchNum):
                #         output_rgb_col.append(output_rgb_patches_best[cnt])
                #         cnt = cnt + 1
                #     output_rgb_new.append(torch.cat(output_rgb_col, dim=-1))
                # output_rgb_new = torch.cat(output_rgb_new, dim=-2)[None, ...]
                # self.output["coarse_raycolor"] = output_rgb_new.permute(0, 2, 3, 1).view(1, -1, 3)
        else:
            raise NotImplementedError
            # import pdb; pdb.set_trace()
            # output_rgb_best = self.output["coarse_raycolor"]
            # B, N, H, W = self.blur_kernels.shape
            # loss_best = self.l2loss(output_rgb_best, self.gt_image)
            #
            # output_original = self.output["coarse_raycolor"]
            # B1, N1, C1 = output_original.shape
            # output_original = output_original.reshape(B1, int(math.sqrt(N1)), int(math.sqrt(N1)), C1).permute(0, 3, 1,
            #                                                                                                   2)
            # c1 = output_original[:, 0:1, :, :]
            # c2 = output_original[:, 1:2, :, :]
            # c3 = output_original[:, 2:3, :, :]
            # mask = torch.ones_like(c1)
            #
            # for i in range(N):
            #     blur_kernal = self.blur_kernels[:, i:i + 1, :, :]
            #     blur_kernal = nn.Parameter(data=blur_kernal, requires_grad=False).cuda()
            #     mask_out = F.conv2d(mask, blur_kernal, padding=int(H / 2))  # avoid the influence of padding.
            #     c1_out = F.conv2d(c1, blur_kernal, padding=int(H / 2)) / mask_out
            #     c2_out = F.conv2d(c2, blur_kernal, padding=int(H / 2)) / mask_out
            #     c3_out = F.conv2d(c3, blur_kernal, padding=int(H / 2)) / mask_out
            #     blur_output = torch.cat([c1_out, c2_out, c3_out], dim=1)
            #
            #     # save image
            #     # img = blur_output.permute(0,2,3,1).detach().cpu().numpy()
            #     # gt = self.gt_image.reshape(B1, int(math.sqrt(N1)), int(math.sqrt(N1)), C1).detach().cpu().numpy()
            #     # img = np.clip(img[0,:,:,:], 0, 1) * 255
            #     # gt = np.clip(gt[0,:,:,:], 0, 1) * 255
            #     # cv2.imwrite('img.png', np.uint8(img))
            #     # cv2.imwrite('gt.png', np.uint8(gt))
            #     # import pdb; pdb.set_trace()
            #
            #     blur_output = blur_output.reshape(B1, C1, -1).permute(0, 2, 1)
            #     loss_tmp = self.l2loss(blur_output, self.gt_image)
            #     if loss_tmp < loss_best:
            #         output_rgb_best = blur_output
            #         loss_best = loss_tmp
            # self.output["coarse_raycolor"] = output_rgb_best

    def learnable_blur_update_output(self, blur_predictor, visualize=False, faster_version=True):
        if self.dilation_PatchNum > 0:
            N = self.dilation_PatchNum * self.dilation_PatchNum  # number of patches
            K_size = self.opt.learnable_blur_kernel_size  # predicted kernel size
            sample_size = self.dilation_PatchNum * self.dilation_PatchSize
            gt_rgb_original = self.gt_image.reshape(1, sample_size, sample_size, -1).permute(0, 3, 1, 2)
            output_rgb_original = self.output["coarse_raycolor"].reshape(1, sample_size, sample_size, -1).permute(0, 3, 1, 2)
            gt_rgb_patches = []
            output_rgb_patches = []

            # reshape 1xCxHxW to BxCxH'xW', H' and W' are small patch sizes.
            xv_patches = self.xv_patches
            yv_patches = self.yv_patches
            if len(xv_patches) == 0 or len(yv_patches) == 0:
                x = np.arange(0, int(sample_size))
                y = np.arange(0, int(sample_size))
                xv, yv = np.meshgrid(x, y)
                # xv_patches, yv_patches = [], []
                for patch_i in range(self.dilation_PatchNum):
                    for patch_j in range(self.dilation_PatchNum):
                        xv_patch = xv[patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                                   patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                        yv_patch = yv[patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                                   patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                        xv_patches.append(xv_patch)
                        yv_patches.append(yv_patch)
                self.xv_patches = np.stack(xv_patches)
                self.yv_patches = np.stack(yv_patches)

            if faster_version:
                time_0 = time.time()
                gt_rgb_patches = gt_rgb_original[0, :, yv_patches, xv_patches].permute(1, 0, 2, 3)
                output_rgb_patches = output_rgb_original[0, :, yv_patches, xv_patches].permute(1, 0, 2, 3)
                time_1 = time.time()
                reshape_time = time_1 - time_0
            else:
                raise NotImplementedError
                # time_0 = time.time()
                # for patch_i in range(self.dilation_PatchNum):
                #     for patch_j in range(self.dilation_PatchNum):
                #         gt_rgb_patch = gt_rgb_original[0, :,
                #                        patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                #                        patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                #         output_rgb_patch = output_rgb_original[0, :,
                #                            patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize,
                #                            patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                #         gt_rgb_patches.append(gt_rgb_patch)
                #         output_rgb_patches.append(output_rgb_patch)
                # gt_rgb_patches = torch.stack(gt_rgb_patches)
                # output_rgb_patches = torch.stack(output_rgb_patches)
                # time_1 = time.time()
                # reshape_time = time_1 - time_0
                # import pdb; pdb.set_trace()

            # currently, use gray patches for kernel prediction.
            if self.opt.learnable_blur_kernel_conv:
                gt_gray_flatten = torch.mean(gt_rgb_patches, axis=1, keepdim=True)
                output_gray_flatten = torch.mean(output_rgb_patches, axis=1, keepdim=True)
                predicted_blur_kernels = blur_predictor[1](blur_predictor[0](torch.cat((gt_gray_flatten, output_gray_flatten), dim=1)).view(N, -1))
            else:
                gt_gray_flatten = torch.mean(gt_rgb_patches, axis=1).view(N, -1)
                output_gray_flatten = torch.mean(output_rgb_patches, axis=1).view(N, -1)
                predicted_blur_kernels = blur_predictor(torch.cat((gt_gray_flatten, output_gray_flatten), dim=-1))
            time_2 = time.time()
            predict_time = time_2 - time_1

            if faster_version:
                # ways normalize predicted blur kernels
                if self.opt.learnable_blur_kernel_norm == 0:
                    blur_kernels = predicted_blur_kernels[:, 0:K_size * K_size].view(N, 1, K_size, K_size)
                    blur_kernels = blur_kernels / torch.sum(blur_kernels, dim=(2, 3), keepdim=True)
                else:
                    blur_kernels = F.softmax(predicted_blur_kernels[:, 0:K_size * K_size], dim=-1).view(N, 1, K_size, K_size)

                # ways learn blur kernels
                if self.opt.learnable_blur_kernel_mode == 0:
                    pass
                elif self.opt.learnable_blur_kernel_mode == 4:
                    predicted_combine_weights = predicted_blur_kernels[:, -1][..., None, None, None]
                    identity_kernels = torch.zeros_like(blur_kernels)  # can be pre-defined
                    identity_kernels[:, :, int(K_size / 2), int(K_size / 2)] = 1.0  # can be pre-defined
                    blur_kernels = predicted_combine_weights * blur_kernels + (1 - predicted_combine_weights) * identity_kernels
                    blur_kernels = blur_kernels / torch.sum(blur_kernels, dim=(2, 3), keepdim=True)
                else:
                    raise NotImplementedError

                masks = torch.ones_like(output_rgb_patches.permute(1, 0, 2, 3))  # can be pre-defined
                # ways handling boundary areas
                if self.opt.boundary_mode == 0:
                    mask_outputs = F.conv2d(masks, blur_kernels, padding=int(K_size / 2), groups=N)
                    blurred_rgb_patches_placeholder = F.conv2d(output_rgb_patches.permute(1, 0, 2, 3), blur_kernels, padding=int(K_size / 2), groups=N) / (mask_outputs + 1e-10)
                elif self.opt.boundary_mode == 1:
                    mask_outputs = F.conv2d(masks, blur_kernels, padding=int(K_size / 2), groups=N)
                    blurred_rgb_patches_placeholder = F.conv2d(output_rgb_patches.permute(1, 0, 2, 3), blur_kernels, padding=int(K_size / 2), groups=N) + (1 - mask_outputs) * output_rgb_patches.permute(1, 0, 2, 3)
                elif self.opt.boundary_mode == 2:
                    mask_outputs = F.conv2d(masks, blur_kernels.clone().detach(), padding=int(K_size / 2), groups=N)
                    blurred_rgb_patches_placeholder = F.conv2d(output_rgb_patches.permute(1, 0, 2, 3), blur_kernels, padding=int(K_size / 2), groups=N) + (1 - mask_outputs) * output_rgb_patches.permute(1, 0, 2, 3)
                else:
                    """
                    The above blur simulation ways cannot correctly handle pixels around patch boundary.
                    To improve:
                    sample a large patch (e.g.,  16x16, split it into 4 8x8 patches).
                    Predict blur kernels on small patches.
                    utilize rendered pixels from adjacent patches when operating around patch boundary.  
                    """
                    raise NotImplementedError
                blurred_rgb_patches_placeholder = blurred_rgb_patches_placeholder.permute(1, 0, 2, 3)
                time_3 = time.time()
                sim_time = time_3 - time_2
            else:
                raise NotImplementedError
                # blurred_rgb_patches_placeholder = torch.zeros_like(output_rgb_patches)
                # mask = torch.ones((1, 3, self.dilation_PatchSize, self.dilation_PatchSize), device=output_rgb_patches.device)
                # mask = mask.permute(1, 0, 2, 3)  # 3x1xH'xW'
                # for i in range(N):
                #     blur_kernel = predicted_blur_kernels[i:i + 1, 0:K_size*K_size]
                #     if self.opt.learnable_blur_kernel_norm == 0:
                #         blur_kernel = (blur_kernel / torch.sum(blur_kernel)).view(1, 1, K_size, K_size)
                #     else:
                #         blur_kernel = F.softmax(blur_kernel, dim=-1).view(1, 1, K_size, K_size)
                #
                #     if self.opt.learnable_blur_kernel_mode == 0:
                #         blur_kernel = blur_kernel
                #     elif self.opt.learnable_blur_kernel_mode == 4:
                #         identity_kernel = torch.zeros_like(blur_kernel)
                #         identity_kernel[:, :, int(K_size/2), int(K_size/2)] = 1.0
                #         combine_weight = predicted_blur_kernels[i:i + 1, -1]
                #         blur_kernel = combine_weight * blur_kernel + (1 - combine_weight) * identity_kernel
                #         blur_kernel = blur_kernel/torch.sum(blur_kernel)
                #     else:
                #         raise NotImplementedError
                #
                #     if self.opt.boundary_mode == 0:
                #         mask_out = F.conv2d(mask, blur_kernel, padding=int(K_size / 2))  # avoid the influence of zero padding.
                #         blur_output = F.conv2d(output_rgb_patches[i:i+1, :, :, :].permute(1, 0, 2, 3), blur_kernel, padding=int(K_size / 2)) / (mask_out + 1e-10)
                #     elif self.opt.boundary_mode == 1:
                #         mask_out = F.conv2d(mask, blur_kernel, padding=int(K_size / 2))
                #         blur_output = F.conv2d(output_rgb_patches[i:i + 1, :, :, :].permute(1, 0, 2, 3), blur_kernel, padding=int(K_size / 2))\
                #                       + (1 - mask_out) * output_rgb_patches[i:i + 1, :, :, :].permute(1, 0, 2, 3)
                #     elif self.opt.boundary_mode == 2:
                #         mask_out = F.conv2d(mask, blur_kernel.clone().detach(), padding=int(K_size / 2))
                #         blur_output = F.conv2d(output_rgb_patches[i:i + 1, :, :, :].permute(1, 0, 2, 3), blur_kernel, padding=int(K_size / 2)) \
                #                       + (1 - mask_out) * output_rgb_patches[i:i + 1, :, :, :].permute(1, 0, 2, 3)
                #     else:
                #         """stop the gradient of blur_kernel, then calculate the mask_output"""
                #         import pdb; pdb.set_trace()
                #
                #     blurred_rgb_patches_placeholder[i:i+1, :, :, :] = blur_output.permute(1, 0, 2, 3)

                #     visualize = False
                #     if visualize:
                #         vis_kernel = blur_kernel.detach().cpu().numpy()[0, 0, :, :]
                #         vis_kernel = cv2.resize(vis_kernel * 255, (32, 32))[..., None]
                #         vis_kernel = np.tile(vis_kernel, (1, 1, 3))
                #         vis_input = output_rgb_patches[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
                #         vis_input = cv2.resize(vis_input*255, (32, 32))
                #         vis_output = blur_output.permute(1, 2, 3, 0).detach().cpu().numpy()[0]
                #         vis_output = cv2.resize(vis_output*255, (32, 32))
                #         vis_gt = gt_rgb_patches[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
                #         vis_gt = cv2.resize(vis_gt*255, (32, 32))
                #         vis_contents = np.concatenate([vis_input, vis_output, vis_gt, vis_kernel], axis=1)
                #         cv2.imwrite('%06d.png' % i, np.uint8(np.minimum(vis_contents, 255)))
                # time_3 = time.time()
                # sim_time = time_3 - time_2
                # import pdb; pdb.set_trace()

            # reshape back
            if faster_version:
                output_rgb_new = []
                # blurred_rgb_patches_list = torch.split(blurred_rgb_patches_placeholder, 1, dim=0)
                blurred_rgb_patches_list = list(torch.chunk(blurred_rgb_patches_placeholder, 1, dim=0)[0])
                for col_j in range(self.dilation_PatchNum):
                    output_rgb_new.append(torch.cat(blurred_rgb_patches_list[col_j*self.dilation_PatchNum:(col_j+1)*self.dilation_PatchNum], dim=-1))
                output_rgb_new = torch.cat(output_rgb_new, dim=-2)[None, ...]
                self.output["coarse_raycolor"] = output_rgb_new.permute(0, 2, 3, 1).view(1, -1, 3)
                time_4 = time.time()
                reshapeback_time = time_4 - time_3
            else:
                raise NotImplementedError
                # cnt = 0
                # output_rgb_new = []
                # for patch_i in range(self.dilation_PatchNum):
                #     output_rgb_col = []
                #     for patch_j in range(self.dilation_PatchNum):
                #         output_rgb_col.append(blurred_rgb_patches_placeholder[cnt])
                #         cnt = cnt + 1
                #     output_rgb_new.append(torch.cat(output_rgb_col, dim=-1))
                # output_rgb_new = torch.cat(output_rgb_new, dim=-2)[None, ...]
                # self.output["coarse_raycolor"] = output_rgb_new.permute(0, 2, 3, 1).view(1, -1, 3)
                # time_4 = time.time()
                # reshapeback_time = time_4 - time_3
                # import pdb; pdb.set_trace()
        else:
            raise NotImplementedError

    def compute_losses(self, color_patch_loss=False, color_patch_match_region=0, combine_two=False, add_ssim_loss=False, detail_aware=False):
        '''
            Compute loss functions.
            The total loss is saved in self.loss_total.
            Every loss will be set to an attr, self.loss_lossname

            Currently, we use L2 loss only. Other patch-related losses are disabled.
        '''

        self.loss_total = 0
        opt = self.opt

        # color losses
        for i, name in enumerate(opt.color_loss_items):
            if name.startswith("ray_masked"):
                if color_patch_loss:
                    raise NotImplementedError
                    # import pdb; pdb.set_trace()
                    # unmasked_name = name[len("ray_masked") + 1:] + '_patch'
                    # patch_channels = self.output[unmasked_name].shape[-1]
                    # masked_output = torch.masked_select(self.output[unmasked_name], (self.output["ray_mask"] > 0)[..., None].expand(-1, -1,patch_channels)).reshape(1, -1, patch_channels)
                    # masked_gt = torch.masked_select(self.gt_image_patch, (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, patch_channels)).reshape(1, -1, patch_channels)
                    # if masked_output.shape[1] > 0:
                    #     loss = self.l2loss(masked_output, masked_gt)
                    # else:
                    #     loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
                elif color_patch_match_region > 0:
                    raise NotImplementedError
                    # import pdb; pdb.set_trace()
                    # unmasked_name = name[len("ray_masked") + 1:] + '_patch'
                    # patch_channels = self.output[unmasked_name].shape[-1]
                    # masked_output = torch.masked_select(self.output[unmasked_name],(self.output["ray_mask"] > 0)[..., None].expand(-1, -1, patch_channels)).reshape(1, -1, patch_channels)
                    # center_px = self.pixel_index[0, :, :, 0].cpu().numpy().astype(np.int32)
                    # center_py = self.pixel_index[0, :, :, 1].cpu().numpy().astype(np.int32)
                    # tl_px = center_px - int(color_patch_match_region / 2)
                    # tl_py = center_py - int(color_patch_match_region / 2)
                    # patch_size = int(math.sqrt(patch_channels / 3))
                    # num_candidates = (color_patch_match_region - patch_size + 1) ** 2
                    # masked_gt_all = torch.zeros((masked_output.shape[0], masked_output.shape[1], masked_output.shape[2], num_candidates), dtype=torch.float32, device=masked_output.device)
                    # cnt = 0
                    # for region_y in range(color_patch_match_region - patch_size + 1):
                    #     for region_x in range(color_patch_match_region - patch_size + 1):
                    #         cur_patch_tl_px = tl_px + region_x
                    #         cur_patch_tl_py = tl_py + region_y
                    #         gt_image_patch = []
                    #         for y_march in range(patch_size):
                    #             for x_march in range(patch_size):
                    #                 cur_patch_px = cur_patch_tl_px + x_march
                    #                 cur_patch_py = cur_patch_tl_py + y_march
                    #                 gt_image_patch_tmp = self.gt_image_full[:, cur_patch_py, cur_patch_px, :]
                    #                 gt_image_patch_tmp = gt_image_patch_tmp.view(1, -1, 3)
                    #                 gt_image_patch.append(gt_image_patch_tmp)
                    #         gt_image_patch = torch.cat(gt_image_patch, dim=-1)
                    #         masked_gt_image_patch = torch.masked_select(gt_image_patch,(self.output["ray_mask"] > 0)[..., None].expand(-1, -1, patch_channels)).reshape(1, -1, patch_channels)
                    #         masked_gt_all[:, :, :, cnt] = masked_gt_image_patch
                    #         cnt = cnt + 1
                    # if masked_output.shape[1] > 0:
                    #     masked_output = masked_output[..., None].expand(-1, -1, -1, num_candidates)
                    #     l2_distance = torch.mean((masked_output - masked_gt_all) ** 2, dim=-2)
                    #     loss_candidate, _ = torch.min(l2_distance, dim=-1)
                    #     loss = torch.mean(loss_candidate)
                    # else:
                    #     loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
                    #
                    # # combine the patch-based loss and the patch-match loss
                    # if combine_two:
                    #     unmasked_name = name[len("ray_masked") + 1:] + '_patch'
                    #     patch_channels = self.output[unmasked_name].shape[-1]
                    #     masked_output = torch.masked_select(self.output[unmasked_name],(self.output["ray_mask"] > 0)[..., None].expand(-1, -1, patch_channels)).reshape(1, -1, patch_channels)
                    #     masked_gt = torch.masked_select(self.gt_image_patch, (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, patch_channels)).reshape(1, -1, patch_channels)
                    #     if masked_output.shape[1] > 0:
                    #         loss1 = self.l2loss(masked_output, masked_gt)
                    #         loss = 0.5 * loss1 + 0.5 * loss
                elif detail_aware and self.is_train and (self.dilation_PatchSize is not None):
                    raise NotImplementedError
                    # import pdb; pdb.set_trace()
                    # unmasked_name = name[len("ray_masked") + 1:]
                    # masked_output = torch.masked_select(self.output[unmasked_name], (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                    # masked_gt = torch.masked_select(self.gt_image, (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                    # # set empty rays as 0
                    # valid_mask = torch.zeros((self.gt_image.shape[0], self.gt_image.shape[1], self.gt_image.shape[2]), dtype=torch.float32, device=self.gt_image.device)
                    # valid_rays = torch.where(self.output["ray_mask"] > 0)
                    # valid_mask[valid_rays[0], valid_rays[1], :] = 1.0
                    # zero_masked_gt = self.gt_image * valid_mask
                    # zero_masked_output = self.output[unmasked_name] * valid_mask
                    # if masked_output.shape[1] > 0:
                    #     detail_aware_weight_map = self.detail_aware_weights(output_rgb_in=zero_masked_output, gt_rgb_in=zero_masked_gt)
                    #     masked_detail_aware_weight_map = torch.masked_select(detail_aware_weight_map,(self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                    #     loss = self.l2loss(masked_output * masked_detail_aware_weight_map, masked_gt * masked_detail_aware_weight_map)
                    # else:
                    #     loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
                else:
                    unmasked_name = name[len("ray_masked") + 1:]
                    masked_output = torch.masked_select(self.output[unmasked_name], (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                    masked_gt = torch.masked_select(self.gt_image, (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                    if masked_output.shape[1] > 0:
                        loss = self.l2loss(masked_output, masked_gt)

                        # patch-based SSIM loss
                        if add_ssim_loss and (self.dilation_PatchSize is not None) and self.is_train:
                            import pdb; pdb.set_trace()
                            # set empty rays as 0
                            valid_mask = torch.zeros((self.gt_image.shape[0], self.gt_image.shape[1], self.gt_image.shape[2]), dtype=torch.float32, device=self.gt_image.device)
                            valid_rays = torch.where(self.output["ray_mask"] > 0)
                            valid_mask[valid_rays[0], valid_rays[1], :] = 1.0
                            zero_masked_gt = self.gt_image * valid_mask
                            zero_masked_output = self.output[unmasked_name] * valid_mask
                            zero_masked_gt = zero_masked_gt.reshape(1, self.dilation_PatchNum * self.dilation_PatchSize, self.dilation_PatchNum * self.dilation_PatchSize, -1).permute(0, 3, 1, 2)
                            zero_masked_output = zero_masked_output.reshape(1, self.dilation_PatchNum * self.dilation_PatchSize, self.dilation_PatchNum * self.dilation_PatchSize, -1).permute(0, 3, 1, 2)
                            # clamp 0~1
                            patch_gt_all = []
                            patch_output_all = []
                            for patch_i in range(self.dilation_PatchNum):  # for loop is slow
                                for patch_j in range(self.dilation_PatchNum):
                                    patch_gt_cur = zero_masked_gt[:, :, patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize, patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                                    patch_output_cur = zero_masked_output[:, :, patch_i * self.dilation_PatchSize:(patch_i + 1) * self.dilation_PatchSize, patch_j * self.dilation_PatchSize:(patch_j + 1) * self.dilation_PatchSize]
                                    patch_gt_all.append(patch_gt_cur[0, :, :, :])
                                    patch_output_all.append(patch_output_cur[0, :, :, :])
                            patch_gt_all = torch.stack(patch_gt_all)
                            patch_output_all = torch.stack(patch_output_all)
                            ssim_loss = 1 - ssim_module(patch_output_all, patch_gt_all)
                            loss = loss + ssim_loss
                    else:
                        loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)

            elif name.startswith("ray_miss"):
                if color_patch_loss:
                    raise NotImplementedError
                elif color_patch_match_region > 0:
                    raise NotImplementedError
                else:
                    unmasked_name = name[len("ray_miss") + 1:]
                    masked_output = torch.masked_select(self.output[unmasked_name], (self.output["ray_mask"] == 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                    masked_gt = torch.masked_select(self.gt_image, (self.output["ray_mask"] == 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                    if masked_output.shape[1] > 0:
                        loss = self.l2loss(masked_output, masked_gt) * masked_gt.shape[1]
                    else:
                        loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)

            elif name.startswith("ray_depth_masked"):
                pixel_xy = self.input["pixel_idx"][0].long()
                ray_depth_mask = self.output["ray_depth_mask"][0][pixel_xy[..., 1], pixel_xy[..., 0]] > 0
                unmasked_name = name[len("ray_depth_masked") + 1:]
                masked_output = torch.masked_select(self.output[unmasked_name], (ray_depth_mask[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
                masked_gt = torch.masked_select(self.gt_image, (ray_depth_mask[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
                loss = self.l2loss(masked_output, masked_gt)

                # print("loss", loss)
                # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_coarse_raycolor")
                # filepath = os.path.join(
                #     "/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
                # csave = torch.zeros((1, 512, 640, 3))
                # ray_masks = (self.output["ray_mask"] > 0).reshape(1, -1)
                # pixel_xy = self.input["pixel_idx"].reshape(1, -1, 2)[ray_masks, :]
                # # print("masked_output", masked_output.shape, pixel_xy.shape)
                # csave[:, pixel_xy[..., 1].long(), pixel_xy[..., 0].long(), :] = masked_output.cpu()
                # img = csave.view(512, 640, 3).detach().numpy()
                # self.save_image(img, filepath)
                #
                # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_gt")
                # filepath = os.path.join(
                #     "/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
                # csave = torch.zeros((1, 512, 640, 3))
                # ray_masks = (self.output["ray_mask"] > 0).reshape(1, -1)
                # pixel_xy = self.input["pixel_idx"].reshape(1, -1, 2)[ray_masks, :]
                # # print("masked_output", masked_output.shape, pixel_xy.shape)
                # csave[:, pixel_xy[..., 1].long(), pixel_xy[..., 0].long(), :] = masked_gt.cpu()
                # img = csave.view(512, 640, 3).detach().numpy()
                # self.save_image(img, filepath)
                # print("psnrkey recal:",mse2psnr(torch.nn.MSELoss().to("cuda")(masked_output, masked_gt)) )
            else:
                if name not in self.output:
                    print(fmt.YELLOW + "No required color loss item: " + name + fmt.END)
                # print("no_mask")
                loss = self.l2loss(self.output[name], self.gt_image)

            self.loss_total += (loss * opt.color_loss_weights[i] + 1e-6)
            # loss.register_hook(lambda grad: print(torch.any(torch.isnan(grad)), grad, opt.color_loss_weights[i]))

            setattr(self, "loss_" + name, loss)
        # print(torch.sum(self.output["ray_mask"]))

        # scale the loss according to image quality
        if self.frame_weight is not None:
            self.loss_total = self.loss_total * self.frame_weight

        # depth losses
        for i, name in enumerate(opt.depth_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required depth loss item: " + name +
                      fmt.END)
            loss = self.l2loss(self.output[name] * self.gt_mask,
                               self.gt_depth * self.gt_mask)
            self.loss_total += loss * opt.depth_loss_weights[i]
            setattr(self, "loss_" + name, loss)

        # background losses
        for i, name in enumerate(opt.bg_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required mask loss item: " + name +
                      fmt.END)
            loss = self.l2loss(self.output[name] * (1 - self.gt_mask),
                               1 - self.gt_mask)
            self.loss_total += loss * opt.bg_loss_weights[i]
            setattr(self, "loss_" + name, loss)

        # zero_one regularization losses
        for i, name in enumerate(opt.zero_one_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required zero_one loss item: " + name +
                      fmt.END)
                # setattr(self, "loss_" + name, torch.zeros([1], device="cuda", dtype=torch.float32))
            else:
                val = torch.clamp(self.output[name], self.opt.zero_epsilon,
                                  1 - self.opt.zero_epsilon)
                # print("self.output[name]",torch.min(self.output[name]), torch.max(self.output[name]))
                loss = torch.mean(torch.log(val) + torch.log(1 - val))
                self.loss_total += loss * opt.zero_one_loss_weights[i]
                setattr(self, "loss_" + name, loss)

        # l2 square regularization losses
        for i, name in enumerate(opt.l2_size_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required l2_size_loss_item : " + name + fmt.END)
            loss = self.l2loss(self.output[name], torch.zeros_like(self.output[name]))
            # print("self.output[name]", self.output[name].shape, loss.shape)
            self.loss_total += loss * opt.l2_size_loss_weights[i]
            setattr(self, "loss_" + name, loss)

        if opt.sparse_loss_weight > 0:
            # weight and conf_coefficient 1, 1134, 40, 8
            if "weight" not in self.output or "conf_coefficient" not in self.output:
                print(fmt.YELLOW + "No required sparse_loss_weight weight or conf_coefficient : " + fmt.END)

            loss = torch.sum(self.output["weight"] * torch.abs(1 - torch.exp(-2 * self.output["conf_coefficient"]))) / (
                        torch.sum(self.output["weight"]) + 1e-6)
            # print("self.output[name]", self.output[name].shape, loss.shape)
            self.output.pop('weight')
            self.output.pop('conf_coefficient')
            self.loss_total += loss * opt.sparse_loss_weight
            setattr(self, "loss_sparse", loss)

        # self.loss_total = Variable(self.loss_total, requires_grad=True)

    def backward(self):
        self.optimizer.zero_grad()
        if self.opt.is_train:
            self.loss_total.backward()
            self.optimizer.step()

    def optimize_parameters(self, backward=True, total_steps=0):
        self.forward()
        self.backward()

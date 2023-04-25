import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import torch.nn.functional as F



DEVICE = 'cuda'


def warp(x, flo, mode='bilinear'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode=mode)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid, mode=mode)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output
    

def load_image(imfile, t_h=480, t_w=640):
    img = np.array(Image.open(imfile)).astype(np.float32)
    img = cv2.resize(img, (t_w, t_h))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def detect_blurry(img_gray, blur_func='mean', k_size=5):
    
    # inhibit the influence of noise
    if blur_func=='gaussian':
        img_blur = cv2.GaussianBlur(img_gray, (k_size, k_size), 0)
    elif blur_func=='mean':
        img_blur = cv2.blur(img_gray, (k_size, k_size))
    else:
        img_blur = img_gray

    # laplacian filter
    edge_laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    edge_laplacian_abs = np.abs(edge_laplacian)
    
    return edge_laplacian, edge_laplacian_abs


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    import math

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + glob.glob(os.path.join(args.path, '*.jpg'))
        ref_images = glob.glob(os.path.join(args.ref_path, '*.png')) + glob.glob(os.path.join(args.ref_path, '*.jpg'))
        images = sorted(images)

        out_dir = args.out_dir
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        step = args.step
        
        scores_cur = []
        scores_ref = []
        num_images = len(images)
        if args.scene_name == 'scene0101_04':    # the last image has no camera pose, remove it.
            num_images = num_images - 1
        for i in range(num_images):

            cur_id = i
            cur_img_name = os.path.join(args.path, '%s.%s' % (int(cur_id), args.file_extension))
            
            if not cur_id%step==0:
                continue

            if not os.path.isfile(cur_img_name):
                break

            ref_id = cur_id + step   
            ref_img_name = os.path.join(args.ref_path, '%s.%s' % (int(ref_id), args.file_extension)) 
            if not os.path.isfile(ref_img_name):
                ref_img_name = os.path.join(args.ref_path, '%s.%s' % (int(cur_id), args.file_extension))

            print('cur_name: %s' % cur_img_name)
            print('ref_name: %s' % ref_img_name)

            imfile1 = cur_img_name
            imfile2 = ref_img_name

            # generate the laplacian edge map.
            img1_gray = np.float64(cv2.resize(cv2.imread(imfile1, 0), (640, 480)))
            img2_gray = np.float64(cv2.resize(cv2.imread(imfile2, 0), (640, 480)))
            img1_edge, img1_edge_abs = detect_blurry(img1_gray) 
            img2_edge, img2_edge_abs = detect_blurry(img2_gray)
            
            # filter out the empty space around image boundary (scannet)
            img1_mask = np.zeros_like(img1_edge)
            img2_mask = np.zeros_like(img2_edge)
            img1_mask[20:-20, 20:-20] = 1
            img2_mask[20:-20, 20:-20] = 1

            # img1_mask[:, :] = 1
            # img2_mask[:, :] = 1


            # try to align two frames.
            image1 = load_image(imfile1, 480, 640)
            image2 = load_image(imfile2, 480, 640)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)  # the shape of flow up: tensor[1, 2, h, w]; 2: (f_x, f_y)  image1_x + f+x == image2_x
            
 
            img2_edge_warped = warp(torch.from_numpy(np.float32(img2_edge[None, None, ...])).cuda(), flow_up, mode='nearest')
            img2_edge_abs_warped = warp(torch.from_numpy(np.float32(img2_edge_abs[None, None, ...])).cuda(), flow_up, mode='nearest')
            img2_mask_warped = warp(torch.from_numpy(np.float32(img2_mask[None, None, ...])).cuda(), flow_up, mode='nearest')
            img2_edge_warped = img2_edge_warped[0, 0, :, :].cpu().numpy()
            img2_edge_abs_warped = img2_edge_abs_warped[0, 0, :, :].cpu().numpy()
            img2_mask_warped = img2_mask_warped[0, 0, :, :].cpu().numpy()

            pixel_mask = img2_mask_warped*img1_mask
            pixel_used = np.where(pixel_mask==1)
            # cv2.imwrite('./mask.jpg', np.uint8(pixel_mask*255))
            
            # calculate the blur score
            img1_blur_score = img1_edge[pixel_used[0], pixel_used[1]].var()
            img2_blur_score = img2_edge_warped[pixel_used[0], pixel_used[1]].var()
            print('blur score: cur_id %s: %s, ref_id %s: %s' % (cur_id, img1_blur_score, ref_id, img2_blur_score))
            
            scores_cur.append(img1_blur_score)
            scores_ref.append(img2_blur_score)

        # align scores, consecutive frames
        scores_absolute = []
        scale = 1
        for i in range(len(scores_cur)):
            cur = scores_cur[i] * scale
            ref = scores_ref[i] * scale
            scores_absolute.append(cur)
            if i==(len(scores_cur) - 1):
                continue
            cur_next = scores_cur[i+1]
            scale = ref/cur_next
            print('id: %s before: %s, %s, after: %s, %s' % (i*step, scores_cur[i], scores_ref[i], cur, ref))
        scores_absolute = np.array(scores_absolute)
        # print(len(scores_absolute))

        # calculate weights
        window_size = args.window_size
        moving = args.sliding_step
        frame_weight = np.zeros(len(scores_absolute))
        frame_count = np.zeros(len(scores_absolute))
        flag = 1
        begin_idx = 0
        while flag:
            end_idx = begin_idx + window_size
            if end_idx >= (len(scores_absolute)):
                end_idx = len(scores_absolute)
                flag=0
            score_bundle = scores_absolute[begin_idx:end_idx]
            mean_bundle = np.mean(score_bundle)
            weight_bundle = score_bundle/mean_bundle
            frame_weight[begin_idx:end_idx] = frame_weight[begin_idx:end_idx] + weight_bundle
            frame_count[begin_idx:end_idx] = frame_count[begin_idx:end_idx] + 1
            begin_idx = begin_idx + moving
        
        frame_weight = frame_weight / frame_count

        for k in range(len(frame_weight)):
            print('%s.jpg: %s' % (k*step, frame_weight[k]))
        
        np.save(out_dir + '/%s_frame_weight_step%s.npy' % (args.scene_name, args.step), frame_weight)
        print('save pre-computed weights to %s' % out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', type=str, default='../data_src/scannet/scans/scene0101_04/exported/color/', help="color image path")
    parser.add_argument('--ref_path',type=str, default='../data_src/scannet/scans/scene0101_04/exported/color/', help="color image path")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--step', type=int, default=5, help="every xxx frame as training frame")
    parser.add_argument('--out_dir', type=str, default='./result/', help="output dir")
    parser.add_argument('--scene_name', type=str, default='scene0101_04', help="the scene name")
    parser.add_argument('--file_extension', type=str, default='jpg', help="extension of file name")
    parser.add_argument('--window_size', type=int, default=10, help="size of sliding window")
    parser.add_argument('--sliding_step', type=int, default=5, help="move xxx frames")
    args = parser.parse_args()

    demo(args)

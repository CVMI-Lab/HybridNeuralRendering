#!/bin/bash

nrCheckpoint="../checkpoints"
nrDataRoot="../data_src"
name='scene241_dilated_uniform_7818_blur_nearest4_deltaView_drop0.5_patch_raypoints_residuals_frameWeight1'  # folder saves pre-trained models

resume_iter=200000 # 20000 #latest

data_root="${nrDataRoot}/scannet/scans/"
scan="scene0241_01"
normview=0
edge_filter=10 # pixels crop out at image edge
point_conf_mode="1" # 0 for only at features, 1 for multi at weight
point_dir_mode="1" # 0 for only at features, 1 for color branch
point_color_mode="1" # 0 for only at features, 1 for color branch

agg_feat_xyz_mode="None"
agg_alpha_xyz_mode="None"
agg_color_xyz_mode="None"
feature_init_method="rand" #"rand" # "zeros"
agg_axis_weight=" 1. 1. 1."
agg_dist_pers=20
radius_limit_scale=4
depth_limit_scale=0
vscale=" 2 2 2 "
kernel_size=" 3 3 3 "
query_size=" 3 3 3 "
vsize=" 0.008 0.008 0.008 " #" 0.005 0.005 0.005 "
wcoord_query=1
z_depth_dim=400
max_o=610000
ranges=" -10.0 -10.0 -10.0 10.0 10.0 10.0 "
SR=24
K=8
P=26
NN=2


act_type="LeakyReLU"

agg_intrp_order=2
agg_distance_kernel="linear" #"avg" #"feat_intrp"
weight_xyz_freq=2
weight_feat_dim=8

point_features_dim=32
shpnt_jitter="passfunc" #"uniform" # uniform gaussian

which_agg_model="viewmlp"
apply_pnt_mask=1
shading_feature_mlp_layer0=1 #2
shading_feature_mlp_layer1=2 #2
shading_feature_mlp_layer2=0 #1
shading_feature_mlp_layer3=2 #1
shading_alpha_mlp_layer=1
shading_color_mlp_layer=4
shading_feature_num=256
dist_xyz_freq=5
num_feat_freqs=3
dist_xyz_deno=0


raydist_mode_unit=1
dataset_name='scannet_ft'
pin_data_in_memory=1
model='mvs_points_volumetric'
near_plane=0.1
far_plane=8.0
which_ray_generation='near_far_linear' #'nerf_near_far_linear' #
domain_size='1'
dir_norm=0

which_tonemap_func="off" #"gamma" #
which_render_func='radiance'
which_blend_func='alpha'
out_channels=4

num_pos_freqs=10
num_viewdir_freqs=4 #6

random_sample='random'
random_sample_size=56 # 32 * 32 = 1024

batch_size=1

plr=0.002
lr=0.0005 # 0.0005 #0.00015
lr_policy="iter_exponential_decay"
lr_decay_iters=1000000
lr_decay_exp=0.1

gpu_ids='0'

checkpoints_dir="${nrCheckpoint}/scannet/"
resume_dir="${nrCheckpoint}/init/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20"

test_num_step=1
visual_items='coarse_raycolor gt_image '
color_loss_weights=" 1.0 0.0 0.0 "
color_loss_items='ray_masked_coarse_raycolor ray_miss_coarse_raycolor coarse_raycolor'
test_color_loss_items='coarse_raycolor ray_miss_coarse_raycolor ray_masked_coarse_raycolor'

bg_color="white" #"0.0,0.0,0.0,1.0,1.0,1.0"
split="train"

use_nearest=4

random_position=1
drop_ratio=0.5
drop_disturb_range=0 #480 640
ray_points=1
drop_patch=1

learn_residuals=1

feature_guidance=1
mixup_mode='partial'
refine_blend=0
dynamic_weight=0
use_delta_view=1

dilation_setup='7_8_1_8'
dilation_mode='uniform'

add_blur_sim=0
blur_kernel_version=3  # 1: asymmetrical, 2: symmetrical, 3: both
blur_kernel_size=9
num_move_dirs=8
move_dists='1,2,4'

# a learnable way
learnable_blur_kernel=0
learnable_blur_kernel_size=9
learnable_blur_kernel_mode=4
# 0: directly predict the blur kernel;
# 4: predicted blur kernel * predicted weight + identity kernel * (1 - predicted weight);

boundary_mode=1

use_frame_weight=0
weight_exp=1
select_high_quality=0


cd run

python3 test_ft.py \
        --name $name \
        --scan $scan \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --model $model \
        --which_render_func $which_render_func \
        --which_blend_func $which_blend_func \
        --out_channels $out_channels \
        --num_pos_freqs $num_pos_freqs \
        --num_viewdir_freqs $num_viewdir_freqs \
        --random_sample $random_sample \
        --random_sample_size $random_sample_size \
        --batch_size $batch_size \
        --gpu_ids $gpu_ids \
        --checkpoints_dir $checkpoints_dir \
        --pin_data_in_memory $pin_data_in_memory \
        --test_num_step $test_num_step \
        --test_color_loss_items $test_color_loss_items \
        --bg_color $bg_color \
        --split $split \
        --which_ray_generation $which_ray_generation \
        --near_plane $near_plane \
        --far_plane $far_plane \
        --dir_norm $dir_norm \
        --which_tonemap_func $which_tonemap_func \
        --resume_dir $resume_dir \
        --resume_iter $resume_iter \
        --feature_init_method $feature_init_method \
        --agg_axis_weight $agg_axis_weight \
        --agg_distance_kernel $agg_distance_kernel \
        --radius_limit_scale $radius_limit_scale \
        --depth_limit_scale $depth_limit_scale  \
        --vscale $vscale    \
        --kernel_size $kernel_size  \
        --SR $SR  \
        --K $K  \
        --P $P \
        --NN $NN \
        --agg_feat_xyz_mode $agg_feat_xyz_mode \
        --agg_alpha_xyz_mode $agg_alpha_xyz_mode \
        --agg_color_xyz_mode $agg_color_xyz_mode  \
        --raydist_mode_unit $raydist_mode_unit  \
        --agg_dist_pers $agg_dist_pers \
        --agg_intrp_order $agg_intrp_order \
        --shading_feature_mlp_layer0 $shading_feature_mlp_layer0 \
        --shading_feature_mlp_layer1 $shading_feature_mlp_layer1 \
        --shading_feature_mlp_layer2 $shading_feature_mlp_layer2 \
        --shading_feature_mlp_layer3 $shading_feature_mlp_layer3 \
        --shading_feature_num $shading_feature_num \
        --dist_xyz_freq $dist_xyz_freq \
        --shpnt_jitter $shpnt_jitter \
        --shading_alpha_mlp_layer $shading_alpha_mlp_layer \
        --shading_color_mlp_layer $shading_color_mlp_layer \
        --which_agg_model $which_agg_model \
        --color_loss_weights $color_loss_weights \
        --num_feat_freqs $num_feat_freqs \
        --dist_xyz_deno $dist_xyz_deno \
        --apply_pnt_mask $apply_pnt_mask \
        --point_features_dim $point_features_dim \
        --color_loss_items $color_loss_items \
        --visual_items $visual_items \
        --act_type $act_type \
        --point_conf_mode $point_conf_mode \
        --point_dir_mode $point_dir_mode \
        --point_color_mode $point_color_mode \
        --normview $normview \
        --edge_filter $edge_filter \
        --vsize $vsize \
        --wcoord_query $wcoord_query \
        --ranges $ranges \
        --z_depth_dim $z_depth_dim \
        --max_o $max_o \
        --query_size $query_size \
        --debug\
        --use_nearest $use_nearest \
        --drop_ratio $drop_ratio \
        --drop_disturb_range $drop_disturb_range \
        --dilation_setup $dilation_setup \
        --dilation_mode $dilation_mode \
        --use_frame_weight $use_frame_weight \
        --add_blur_sim $add_blur_sim \
        --learn_residuals $learn_residuals \
        --mixup_mode $mixup_mode \
        --random_position $random_position \
        --refine_blend $refine_blend \
        --dynamic_weight $dynamic_weight \
        --feature_guidance $feature_guidance\
        --use_delta_view $use_delta_view\
        --ray_points $ray_points\
        --blur_kernel_version $blur_kernel_version\
        --blur_kernel_size $blur_kernel_size\
        --weight_exp $weight_exp\
        --learnable_blur_kernel $learnable_blur_kernel\
        --learnable_blur_kernel_size $learnable_blur_kernel_size\
        --learnable_blur_kernel_mode $learnable_blur_kernel_mode\
        --drop_patch $drop_patch\
        --boundary_mode $boundary_mode\
        --num_move_dirs $num_move_dirs\
        --move_dists $move_dists\
        --select_high_quality $select_high_quality

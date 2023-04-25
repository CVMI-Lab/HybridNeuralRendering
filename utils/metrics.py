"""
Recalculate metrics excluding training views
"""

import math
exp_names = ['scene101_random_nearest4_deltaView_drop0.5_raypoints_residuals']
root ='/mnt/proj74/wenboli/pdai/HybridNeuralRendering/checkpoints/scannet/'
step = 5  # training steps
# exp_names = ['chair_random_nearest4_deltaView_drop0.5_raypoints_residuals']
# root ='/mnt/proj74/wenboli/pdai/hybrid_neural_rendering/checkpoints/nerfsynth/'
# step = 10000  # training steps
metrics = ['lpips', 'ssim', 'psnr']

for i in range(len(exp_names)):
    exp_name = exp_names[i]
    for metric in metrics:
        f = open(root+exp_name+'/test_200000/images/%s.txt' % metric, 'r')
        lines = f.readlines()
        score_sum = 0
        cnt = 0
        idx = 0
        for line in lines:
            line = line.strip()
            if idx % step == 0:
                idx = idx + 1
                continue
            score = float(line[0:5])
            if float(line[-1]) >= 1 and line[-3] == '-':
                score = score/(10**float(line[-1]))
            if float(line[-1]) >= 1 and line[-3] == '+':
                score = score*(10**float(line[-1]))
            score_sum = score_sum + score
            cnt = cnt + 1
            idx = idx + 1
        score_mean = score_sum/cnt
        print('exp_name: %s, metric: %s, num_samples: %s, mean: %s' % (exp_name, metric, cnt, score_mean))
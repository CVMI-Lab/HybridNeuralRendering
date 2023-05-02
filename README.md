# Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur

**Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur** (CVPR 2023)  
[Peng Dai*](https://github.com/daipengwa), [Yinda Zhang*](https://www.zhangyinda.com/), [Xin Yu](https://scholar.google.com/citations?user=JX8kSoEAAAAJ&hl=zh-CN), [Xiaoyang Lyu](https://scholar.google.com/citations?user=SF7bq48AAAAJ&hl=zh-CN), [Xiaojuan Qi](https://scholar.google.com/citations?user=bGn0uacAAAAJ&hl=en).
<br>[Paper](https://arxiv.org/abs/2304.12652), [Project_page](https://daipengwa.github.io/Hybrid-Rendering-ProjectPage/)


## Introduction
<img src='./images/pipeline.png' width=1000>
<br>
Our method takes advantages of both neural 3D representation and image-based rendering to render high-fidelity and temporally consistent results. 
Specifically, the image-based features compensate for the defective neural 3D features, and the neural 3D features boost the temporal consistency of image-based features.
Moreover, we propose efficient designs to handle motion blurs that occur during capture.

## Environment
* We use the same environment as PointNeRF, please follow their [installation](https://github.com/Xharlie/pointnerf) step by step. (conda virtual environment is recommended) 

* Install the dependent python libraries
```Shell
pip install opencv_python imutils
```

The code has been tested on a single NVIDIA 3090 GPU. 


## Preparation

* Please download [datasets](https://www.dropbox.com/s/hwcymldycf3z87y/data_src.zip?dl=0) used in this paper. The layout looks like this:
```
HybridNeuralRendering
├── data_src
    ├── scannet
    │   │──frame_weights_step5 
    │   │──scans 
    |   │   │──scene0101_04
    │   │   │──scene0241_01
    │   │   │──livingroom
    │   │   │──vangoroom
    ├── nerf
    │   │──nerf_synthetic
    │   │   │──chair
    │   │   │──lego
```

* Download [pre-trained models](https://www.dropbox.com/sh/1v0p7bnhrixa6bs/AAABuWyTkfdFDOe6vZ1IZheZa?dl=0). Since we currently focus on per-scene optimization, make sure that "checkpoints" folder contains "init" and "MVSNet" folders with pre-trained models. 

## Quality-aware weights
The weights have been included in the "frame_weights_step5" folder. Alternatively, you can follow the [RAFT](https://github.com/princeton-vl/RAFT) to build the running environment and download their pre-trained models. Then, compute quality-aware weights by running:
```Shell
cd raft
python demo_content_aware_weights.py --model=models/raft-things.pth --path=path of RGB images  --ref_path=path of RGB images  --scene_name=scene name
```

## Train
We take the training on ScanNet 'scene0241_01' for example (The training scripts will resume training if "xxx.pth" files are provided in the pre-trained scene folder, e.g., "checkpoints/scannet/xxx/xxx.pth". Otherwise, train from scratch.):

### Hybrid rendering
Only use hybrid rendering, run:
```Shell
bash ./dev_scripts/w_scannet_etf/scene241_hybrid.sh
```

### Hybrid rendering + blur-handling module (pre-defined degradation kernels)
The full version of our method, run:
```Shell
bash ./dev_scripts/w_scannet_etf/scene241_full.sh
```

### Hybrid rendering + blur-handling module (learned degradation kernels)
Instead of using pre-defined kernels, we also provide an efficient way to estimate degradation kernels from rendered and GT patches. Specifically, flattened rendering and GT patches are concatenated and fed into an MLP to predict the degradation kernel. 
```Shell
bash ./dev_scripts/w_scannet_etf/scene241_learnable.sh
```

## Evaluation
We take the evaluation on ScanNet 'scene0241_01' for example:</br>
Please specify "name" in "scene241_test.sh" to evaluate different experiments, then run:
```Shell
bash ./dev_scripts/w_scannet_etf/scene241_test.sh
```
You can directly evaluate using our [pre-trained models](https://www.dropbox.com/sh/1v0p7bnhrixa6bs/AAABuWyTkfdFDOe6vZ1IZheZa?dl=0).

## Results
Our method generates high-fidelity results when comparing with PointNeRF' results and reference images.
Please visit our [project_page](https://daipengwa.github.io/Hybrid-Rendering-ProjectPage/) for more comparisons.
</br>
</br>
<img src='./images/result.png' width=1000>

## Contact
If you have questions, you can email me (daipeng@eee.hku.hk).

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@inproceedings{dai2023hybrid,
  title={Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur},
  author={Dai, Peng and Zhang, Yinda and Yu, Xin and Lyu, Xiaoyang and Qi, Xiaojuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

# Acknowledgement
This repo is heavily based on [PointNeRF](https://github.com/Xharlie/pointnerf) and [RAFT](https://github.com/princeton-vl/RAFT), we thank authors for their brilliant works.

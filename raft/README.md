Please follow the [RAFT](https://github.com/princeton-vl/RAFT) to build the running environment and download the pre-trained model.

To compute quality-aware weights, run
```Shell
python demo_pointnerf_train.py --model=models/raft-things.pth --path=path of RGB images  --ref_path=path of RGB images  --scene_name=scene name
```


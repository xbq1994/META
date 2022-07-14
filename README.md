# META
[ECCV 2022] Mimic Embedding via Adaptive Aggregation: Learning Generalizable Person Re-identification

### Update
2022-07-14: Update Code. 

### Datasets
* Requirements: Market1501, CUHK03, CUHK-SYSU, MSMT17_v2
* 
Please put all the datasets in one directory, and add the path of the directory to DATASETS.PATH in ```projects/META/configs/Base-cnn.yml```

### Installation
* Please see [fast-reid](http://arxiv.org/abs/2008.08528) for fast-reid installation
* Please see [Apex](https://github.com/NVIDIA/apex) for Apex installation

### Train
1. `cd` to folder:
```
 cd projects/Black_reid
```
2. If you want to train with 1-GPU, run:
```
CUDA_VISIBLE_DEVICES=0 train_net.py --config-file= "configs/HAA_baseline_blackreid.yml"
```
   if you want to train with 4-GPU, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 train_net.py --config-file= "configs/HAA_baseline_blackreid.yml"
```

### Evaluation
To evaluate a model's performance, use:
```
CUDA_VISIBLE_DEVICES=0 train_net.py --config-file= "configs/HAA_baseline_blackreid.yml" --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Contacts
If you have any question about the project, please feel free to contact me.

E-mail: boqiang.xu@cripac.ia.ac.cn

## ACKNOWLEDGEMENTS
The code was developed based on the ’fast-reid’ toolbox https://github.com/JDAI-CV/fast-reid.

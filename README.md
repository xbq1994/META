# META
[ECCV 2022] Mimic Embedding via Adaptive Aggregation: Learning Generalizable Person Re-identification [[paper]](https://arxiv.org/pdf/2112.08684.pdf)

### Update
2022-07-14: Update Code. 

### Datasets
* Requirements: Market1501, CUHK03, CUHK-SYSU, MSMT17_v2

Please put all the datasets in one directory, and modify DATASETS in ```projects/META/configs/Base-cnn.yml```

### Installation
* Please check [fast-reid](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md) for fast-reid installation
* Please check [Apex](https://github.com/NVIDIA/apex) for Apex installation
* Compile with cython to accelerate evalution: 
```
bash cd fastreid/evaluation/rank_cylib; make all
```

### Train
If you want to train with 1-GPU, run:
```
python projects/META/train_net.py --config-file projects/META/configs/r50.yml MODEL.DEVICE "cuda:0"
```
If you want to train with 4-GPU, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/META/train_net.py --config-file projects/META/configs/r50.yml --num-gpus 4
```
You can get the results in our paper by training with 4-GPU, please modify SOLVER.IMS_PER_BATCH in ```projects/META/configs/Base-cnn.yml``` (64 for 1-GPU and 256 for 4-GPU)

### Evaluation
To evaluate a model's performance, use:
```
python projects/META/train_net.py --config-file projects/META/configs/r50.yml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```

## Contacts
If you have any question about the project, please feel free to contact me.

E-mail: boqiang.xu@cripac.ia.ac.cn

## ACKNOWLEDGEMENTS
The code was developed based on the ’fast-reid’ toolbox https://github.com/JDAI-CV/fast-reid.

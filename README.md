# Domain Generalization-Aware Uncertainty Introspective Learning for 3D Point Clouds Segmentation


This repository contains the official implementation of our paper **"Domain Generalization-Aware Uncertainty Introspective Learning for 3D Point Clouds Segmentation"**, published in Proceedings of the 32nd ACM International Conference on Multimedia  (ACMMM2024). [Paper Link](https://dl.acm.org/doi/pdf/10.1145/3664647.3681574) 

## Overview
Domain generalization 3D segmentation aims to learn the point clouds with unknown distributions. Feature augmentation has been proven to be effective for domain generalization. However, each point of the 3D segmentation scene contains uncertainty in the target domain, which affects model generalization. This paper proposes the Domain Generalization-Aware Uncertainty Introspective Learning (DGUIL) method, including Potential Uncertainty Modeling (PUM) and Momentum Introspective Learning (MIL), to deal with the point uncertainty in domain shift. Specifically, PUM explores the underlying uncertain point cloud features and generates the different distributions for each point. The PUM enhances the point features over an adaptive range, which provides various information for simulating the distribution of the target domain. Then, MIL is designed to learn generalized feature representation in uncertain distributions. The MIL utilizes uncertainty correlation representation to measure the predicted divergence of knowledge accumulation, which learns to carefully judge and understand divergence through uncertainty introspection loss. Finally, extensive experiments verify the advantages of the proposed method over current state-of-the-art methods.

## Setup Environment
- Python 3.8
- CUDA 11.6
- Pytorch 1.13.0, 
- TorchSparse 2.0.0b0

### Installation Steps

1. **Setting up a Conda Environment**:  
   We recommend establishing a new conda environment for this installation.
```
$ conda create -n pointdr python=3.8
$ conda activate pointdr
```
2. **Installing PyTorch**:  
Install PyTorch, TorchVision with specific CUDA support.
```
$ pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
3. **Additional Dependencies**:  
Install additional utilities and dependencies.
```
$ pip install tqdm
$ sudo apt-get update
$ sudo apt-get install libsparsehash-dev
$ conda install backports
```
4. **Installing TorchSparse**:  
Update and install TorchSparse from its GitHub repository.
```
$ pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```
#### Pip/Venv/Conda
In your virtual environment follow [TorchSparse](https://github.com/mit-han-lab/spvnas). This will install all the base packages.


## Data preparation

#### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
```

## SemanticSTF dataset
Download SemanticSTF dataset from [GoogleDrive](https://forms.gle/oBAkVJeFKNjpYgDA9), [BaiduYun](https://pan.baidu.com/s/10QqPZuzPclURZ6Niv1ch1g)(code: 6haz). Data folders are as follows:
The data should be organized in the following format:
```
/SemanticSTF/
  └── train/
    └── velodyne
      └── 000000.bin
      ├── 000001.bin
      ...
    └── labels
      └── 000000.label
      ├── 000001.label
      ...
  └── val/
      ...
  └── test/
      ...
  ...
  └── semanticstf.yaml
```


## Training
To start training on SemanticKITTI->SemanticSTF, run:
```bash
python train.py configs/kitti2stf/minkunet/DGUIL.yaml --run_dir ./runs/DGUIL/
```

## Evaluation
To evaluate the model, run:
```bash
python evaluate.py configs/kitti2stf/minkunet/DGUIL.yaml --checkpoint_path ./runs/DGUIL/max-iou-test.pt
```

## Checkpoints
We provide the checkpoints trained on SemanticKITTI->SemanticSTF. Download the model weights from the link: [Checkpoint Model](https://pan.baidu.com/s/1HF78YdB0r-VVI7wXqTxrmQ?pwd=krgq). And placer the model checkpoint in the checkpoint directory, run:

```bash
python evaluate.py configs/kitti2stf/minkunet/DGUIL.yaml --checkpoint_path ./checkpoint/DGUIL.pt
```

If you want to evaluate DGUIL_minkunet34, please download the model weights from the [link](https://pan.baidu.com/s/1c_k6pCNtcwxQWE6XMJ-qkQ?pwd=kwsk). And placer the model checkpoint in the checkpoint directory, run:

```bash
python evaluate.py configs/kitti2stf/minkunet/DGUIL_minkunet34.yaml --checkpoint_path ./checkpoint/DGUIL_minkunet34.pt
```

## Acknowledgements
The code is based on the following open-source projects [SemanticSTF](https://github.com/xiaoaoran/SemanticSTF). We thank their authors for making the source code publicly available.

## License
This project is released under the MIT License.

<p align="center">
  <h2 align="center"> <img src="https://github.com/ranrhuang/ranrhuang.github.io/raw/master/spfsplatv2/static/image/icon_v3.png" width="20" style="position: relative; top: 1px;"> SPFSplatV2  <br> Efficient Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views </h2>
 <p align="center">
    <a href="https://ranrhuang.github.io/">Ranran Huang</a>
    ·
    <a href="https://www.imperial.ac.uk/people/k.mikolajczyk">Krystian Mikolajczyk</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2509.17246">Paper</a> | <a href="https://ranrhuang.github.io/spfsplatv2/">Project Page</a>  </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/ranrhuang/ranrhuang.github.io/raw/master/spfsplatv2/static/image/framework_v6.svg" alt="Teaser" width="90%">
  </a>
</p>


<p align="center">
<strong>SPFSplatV2</strong> efficiently leverages masked attention to predict target poses while simultaneously predicting 3D Gaussians from unposed sparse images, without requiring ground-truth poses during either training or inference. 


<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a>
    </li>
    <li>
      <a href="#camera-conventions">Camera Conventions</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#running-the-code">Running the Code</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
</ol>
</details>

## Installation

1. Clone SPFSplat.
```bash
git clone git@github.com:ranrhuang/SPFSplatV2.git
cd SPFSplatV2
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n spfsplatv2 python=3.11
conda activate spfpslatv2
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
cd src/model/encoder/backbone/croco/curope/
python setup.py build_ext --inplace
cd ../../../../../..
```

## Pre-trained Checkpoints
Our models are hosted on [Hugging Face](https://huggingface.co/RanranHuang/SPFSplat) 🤗

|                                                    Model name                                                    | Training resolutions | Training data | Training settings |
|:----------------------------------------------------------------------------------------------------------------:|:--------------------:|:-------------:|:-------------:|
|                 [re10k_spfsplatv2.ckpt]( https://huggingface.co/RanranHuang/SPFSplatV2/resolve/main/re10k_spfsplatv2.ckpt)                  |        256x256       |     re10k     | RE10K, 2 views,  MASt3R-based|
|                  [acid_spfsplatv2.ckpt]( https://huggingface.co/RanranHuang/SPFSplatV2/resolve/main/acid_spfsplatv2.ckpt )                  |        256x256       |     acid      | ACID, 2 views, MASt3R-based |
|                 [re10k_spfsplatv2l.ckpt]( https://huggingface.co/RanranHuang/SPFSplatV2/resolve/main/re10k_spfsplatv2l.ckpt)                  |        256x256       |     re10k     | RE10K, 2 views, VGGT-based |
|                  [acid_spfsplatv2l.ckpt]( https://huggingface.co/RanranHuang/SPFSplatV2/resolve/main/acid_spfsplatv2l.ckpt )                  |        256x256       |     acid      | ACID, 2 views, VGGT-based |

We assume the downloaded weights are located in the `pretrained_weights` directory.



## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## Running the Code
### Training
1. If using MASt3R-based architecture, download the [MASt3R](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) pretrained model and put it in the `./pretrained_weights` directory.

2. Train with:

```bash
# 2 view on MASt3R-based architecture
python -m src.main +experiment=spfsplatv2/re10k wandb.mode=online wandb.name=re10k_spfsplatv2


# 2 view on VGGT-based architecture
python -m src.main +experiment=spfsplatv2-l/re10k wandb.mode=online wandb.name=re10k_spfsplatv2l

```

### Evaluation
#### Novel View Synthesis and Pose Estimation
```bash
# RealEstate10K on MASt3R-based architecture(enable test.align_pose=true if using evaluation-time pose alignment)
python -m src.main +experiment=spfsplatv2/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./pretrained_weights/re10k_spfsplatv2.ckpt \
    test.save_image=true test.align_pose=false

# ACID on MASt3R-based architecture(enable test.align_pose=true if using evaluation-time pose alignment)
python -m src.main +experiment=spfsplatv2/acid mode=test wandb.name=acid \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
  checkpointing.load=./pretrained_weights/acid_spfsplatv2.ckpt \
  test.save_image=false test.align_pose=false

# RealEstate10K on VGGT-based architecture(enable test.align_pose=true if using evaluation-time pose alignment)
python -m src.main +experiment=spfsplatv2-l/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./pretrained_weights/re10k_spfsplatv2l.ckpt \
    test.save_image=true test.align_pose=false

# ACID on VGGT-based architecture(enable test.align_pose=true if using evaluation-time pose alignment)
python -m src.main +experiment=spfsplatv2-l/acid mode=test wandb.name=acid \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
  checkpointing.load=./pretrained_weights/acid_spfsplatv2l.ckpt \
  test.save_image=false test.align_pose=false
    
```


## Camera Conventions
We follow the [pixelSplat](https://github.com/dcharatan/pixelsplat) camera system. The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).
The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Acknowledgements
This project is built upon these excellent repositories: [NoPoSplat](https://github.com/cvg/NoPoSplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [DUSt3R](https://github.com/naver/dust3r), and [CroCo](https://github.com/naver/croco). We thank the original authors for their excellent work.


## Citation

```
@article{huang2025spfsplatv2,
      title={SPFSplatV2: Efficient Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views},
      author={Huang, Ranran and Mikolajczyk, Krystian},
      journal={arXiv preprint arXiv: 2509.17246},
      year={2025}
}
```
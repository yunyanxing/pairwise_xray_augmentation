# Adversarial Pulmonary Pathology Translation for Pairwise Chest X-ray Data Augmentation


This repository contains a Pytorch implementation of the MICCAI 2019 paper "Adversarial Pulmonary Pathology Translation for Pairwise Chest X-ray Data Augmentation".  

Y. Xing, Z. Ge,  R. Zeng, D. Mahapatra, J. Seah, M. Law and T. Drummond, "Adversarial Pulmonary Pathology Translation for Pairwise Chest X-ray Data Augmentation", *International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)*, 2019.

The code is built upon the pytorch implementation of [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix "Pix2Pix").

## Requirements
- Linux or macOS
- Python 3
- PyTorch 0.41+
- CPU or NVIDIA GPU + CUDA CuDNN

## Usage

### Download code

```bash
git clone https://github.com/yunyanxing/pairwise_xray_augmentation.git
cd pairwise_xray_augmentation
```

### Data

Pre-processed X-ray images used in the paper can be found by clicking [here](https://drive.google.com/drive/folders/1z_mKi75LsthvwcXRsVZxZLBR0n3uRchp?usp=sharing "here").

Unzip this folder into datasets/dataset:
```bash
unzip Preprocessed_Images.zip -d datasets/dataset
```

If you wish to pre-process your own data, use datasets/generate_paired_images.py to create bounding box only images (with original images and bounding box labels) and datasets/combine_A_and_B.py to pair bounding box images with original images for input to the Pix2Pix model. This works for data arranged in the same style as the [NIH dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC "NIH dataset").

### Training

Run the following to train (set GPU ID to desired ID, or -1 for CPU):

```bash
python3 train.py --dataroot ./datasets/dataset/Combined --name xray_pix2pix --model pix2pix --direction AtoB --gpu_ids 0
```

Models and example images are saved throughout training in the checkpoints/xray_pix2pix directory.

### Generate Fake Images

Following training, to generate fake images run:
```bash
python3 test.py --dataroot ./datasets/dataset/Combined --name xray_pix2pix --model pix2pix --direction AtoB --gpu_ids 0
```

Images will be saved in the checkpoints/xray_pix2pix/fake_images directory.

## Cites

If you use this code, please cite our paper:
```
@inproceedings{xing2019pairwise,
title={Adversarial Pulmonary Pathology Translation for Pairwise Chest X-ray Data   Augmentation},
author={Xing, Yunyan and Ge, Zongyuan and Zeng, Rui and Mahapatra, Dwarikanath and Seah, Jarrel and Law, Meng and Drummond, Tom},
booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
year={2019}
}
```
Please also cite the Pix2Pix paper (which our code is built upon):
```
@inproceedings{isola2017image,
title={Image-to-Image Translation with Conditional Adversarial Networks},
author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
year={2017}
}
```





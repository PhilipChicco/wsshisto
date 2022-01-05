# Weakly Supervised Segmentation on Neural Compressed Histopathology

### Repo-Updates

This repository is under continous updates.

## Environment Requirements

* Ubuntu 20
* Python 3.6/3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [Pytorch 1.7.1](https://pytorch.org)

## Conda environment installation

````bash
conda env create --name wsshisto python=3.6
conda activate wsshisto
```
````

* run `pip install -r requirements.txt`

## Getting started

### Datasets

* Download WSI daatasets e.g. [Camelyon16](https://camelyon17.grand-challenge.org/)
* Place datasets in appropriate locations.
* Preprocess WSI to generate  WSI thumbnails (normal/tumor) e.g. save as '001_tissue_fig.png', tumor mask from 'xml' annotations (001_tumor.png) all in a single folder (e.g. /cm16/WSI_ALL)
* Refer to [CLAM](https://github.com/mahmoodlab/CLAM) & [DSMIL](https://github.com/binli123/dsmil-wsi) for pre-processing.
* Extract patches from all training WSIs. Save patches and normalize (e.g. save in cm16/train/patches_norm/train/0). The folder (patches_norm/train/0) contains WSIs as folders (001/xx_patch.png .... xx_patch..png), each containing all the patches for that slide.

## Train SimCLR encoder and Compress WSIs

* Train the patch encoder  (see configs/patch_enc.yml)
* run `python graph_research/train.py --config /path/to/patch_enc.yml`
* To compress WSIs using the trained encoder refer to [Tellez et al. NIC](https://github.com/davidtellez/neural-image-compression/blob/master/source/nic/featurize_wsi.py)
* Save the compressed WSIs in a folder i.e., (/path/to/compressed/all/wsi/001.npy) as well as the masks (/path/to/compressed/all/label/001.npy)

## Train | Test WSS

* Train/test the benchmark UNet
* run `python graph_research/{train/test}.py --config ./graph_research/configs/wsi/neu_seg/wsi_seg.yml`
* Train/test proposed WSS-SS
* run `python graph_research/{train/test}.py --config ./graph_research/configs/wsi/neu_seg/wsi_ss_mask.yml`

## Models

* To see more details on the proposed algorithm, refer to models/pooling/seam.py and trainers/wsi_ss_trainer.py

## References 

Our implementation builds upon several existing publicly available code. 

* [JointMIL - MICCAI'20](https://github.com/PhilipChicco/MICCAI2020mil)
* [SEAM - CVPR'20'](https://github.com/YudeWang/SEAM)
* [NIC-TPAMI](https://github.com/davidtellez/neural-image-compression)

# PFFNet: Instance Segmentation in Biomedical and Biological Images


In this project, we proposed a Panoptic Feature Fusion Net (PFFNet) for instance segmentation in biomedical and biological images.



The implementations are for our previous two papers:

[Nuclei Segmentation via a Deep Panoptic Model with Semantic Feature Fusion](https://www.ijcai.org/Proceedings/2019/0121.pdf), IJCAI, 2019.
 
[Panoptic Feature Fusion Net: A Novel Instance Segmentation Paradigm for Biomedical and Biological Images](https://ieeexplore.ieee.org/abstract/document/9325955), IEEE Transactions on Image Processing.



## Introduction and Installation

Please follow [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) to set up the environment. In this project, the Pytorch Version 1.3.1 and CUDA 10.1 are used.


## Data

### Data Introduction

In this work, we use four datasets:

Histopathology Images: TCGA-Kumar, and TNBC. Please download them from [link](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK).

Fluorescence Microscopy Images: BBC039 Version 1. Download from this [link](https://bbbc.broadinstitute.org/BBBC039).

Plant Phenotyping Images: CVPPP 2019. Download the data following their official [site](https://competitions.codalab.org/competitions/18405). 

**If you are using these datasets in your research, please also remember to cite their original work.**

### Data preparation

All the data should be put in `./datasets`. For the detailed path of each dataset, please refer to:

`./maskrcnn_benchmark/config/path_catalog.py`

Here we provide some sample images on TCGA-Kumar experiment (cell-tcga).

Note that the instance annotations are stored in .json files following the MSCOCO format. If you want to generate the annotations by yourself, please follow this [repository](https://github.com/waspinator/pycococreator).

## Model training

For training the model on different datasets in our papers, please refer to:

`./train_gn_tcga.sh, ./train_gn_tnbc.sh, ./train_gn_fluo.sh, and ./train_gn_cvppp.sh` .

## Model inference and Evaluation

Please refer to `./inference` for the code on this part.

To visualize the instance-level mask annotations/predictions, please run `python color_instance.py`.

### TCGA-Kumar

For the experiment on TCGA-Kumar, we rename the testing images and the details are in [Link](https://cloudstor.aarnet.edu.au/plus/s/Tpd3d6H2XxUlkl4).

To get the instance segmentation predictions, run `python tcga_infer.py`. Remember to manually set the path of the pre-trained weights, testing images, and output folder.

To evaluate the segmentation performance under AJI, pixel-f1, and Panoptic Quality (PQ), please run `python tcga_eva.py`. The overall results for all the testing images will be saved in a .xls file.

### TNBC

Our experiments on TNBC dataset is in 3-fold (random) cross-validation. Please refer to `python tnbc_infer_cv3.py` for details on how to name the root for the models weights and predictions.

For evaluation, run `python tnbc_eva_cv3.py`

### BBBC039V1 (fluo)

For instance segmentation predictions, run `python fluo_infer.py`. For evaluation, run `python fluo_eva.py`.

### CVPPP

For 3-fold cross-validation, run `python cvppp_infer_cv3.py` for inference. 

If you would like to participate in the official [challenge](https://competitions.codalab.org/competitions/18405), you might need to submit your predictions in .h5 format. 
Running `python cvppp_tiff_to_h5.py` will help transfer the instance segmentation predictions into the required .h5 format. 

## Citations (Bibtex)
Please consider citing our papers in your publications if they are helpful to your research:
```
@article{liu2021panoptic,
  title={Panoptic Feature Fusion Net: A Novel Instance Segmentation Paradigm for Biomedical and Biological Images},
  author={Liu, Dongnan and Zhang, Donghao and Song, Yang and Huang, Heng and Cai, Weidong},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={2045--2059},
  year={2021},
  publisher={IEEE}
}

```

```
@inproceedings{liu2019nuclei,
  title={Nuclei Segmentation via a Deep Panoptic Model with Semantic Feature Fusion.},
  author={Liu, Dongnan and Zhang, Donghao and Song, Yang and Zhang, Chaoyi and Zhang, Fan and O'Donnell, Lauren and Cai, Weidong},
  booktitle={IJCAI},
  pages={861--868},
  year={2019}
}


```

 
## Thanks to the Third Party Repositories

[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

[maskscoring_rcnn](https://github.com/zjhuang22/maskscoring_rcnn)

[hover_net](https://github.com/vqdang/hover_net)



## Contact

Please contact Dongnan Liu (dongnanliu0201@gmail.com) for any questions.


## License

PFFNet is released under the MIT license. See [LICENSE](LICENSE) for additional details.



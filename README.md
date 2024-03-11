#                           SFOD: Spiking Fusion Object Detection 

![Optional image alt text](figure.png)



This is the official implementation of the 'SFOD: Spiking Fusion Object Detector' .

## Requirements

<p align="center">

|    Repository     | Version |
| :---------------: | :-----: |
|       CUDA        |  11.7   |
|       cuDNN       | V8.0.0  |
|      Python       |  3.9.0  |
|      Pytorch      | 1.12.1  |
|    Torchvision    | 0.13.1  |
|   Torchmetrics    | 0.11.4  |
| Pytorch-lightning |  2.0.1  |
|   SpikingJelly    | 0.0.12  |

</p>

## Pretrained Checkpoints

We will provide the trained models in the pretrained folder, which will include pretrained backbone networks and pretrained SFOD.

## Required Data

To evaluate or train SFOD you will need to download the datasets:

| Dataset Name  |                             Link                             |
| :-----------: | :----------------------------------------------------------: |
| NCARS Dataset | [Download N-CARS Car Classification Dataset &#124; PROPHESEE](https://www.prophesee.ai/2018/03/13/dataset-n-cars/) |
| GEN1 Dataset  | [Download Gen1 Automotive Detection Dataset &#124; PROPHESEE](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) |

## Training

### Training for Backbone

python classification.py -devices auto -num_workers 8 -test -save_ckpt -model densenet-121_16 -loss_fun mse -encoding fre -early_stopping 

python classification.py -devices auto -num_workers 8 -test -save_ckpt -model densenet-121_24 -loss_fun mse -encoding fre -early_stopping 

### Training for SFOD

python object_detection.py -devices auto -num_workers 4 -test -save_ckpt -backbone densenet-121_24 -pretrained_backbone ./pretrained/DenseNet121-24.ckpt -b 16 -fusion -fusion_layers 3 -mode res

## Evaluation

When you perform evaluation, the corresponding pretrained model data needs to be replaced in the appropriate root folder.

### Evaluation for Backbone

 python classification.py -devices auto -num_workers 8 -test -no_train -model densenet-121_16 -loss_fun mse -encoding fre -pretrained DenseNet121-16.ckpt

 python classification.py -devices auto -num_workers 8 -test -no_train -model densenet-121_24 -loss_fun mse -encoding fre -pretrained DenseNet121-24.ckpt

### Evaluation for SFOD

python object_detection.py -num_workers 4 -test -no_train -pretrained SFOD.ckpt -backbone densenet-121_24 -fusion -fusion_layers 3 -mode res

### Code Acknowledgments

This code is based on [object-detection-with-spiking-neural-networks](https://github.com/loiccordone/object-detection-with-spiking-neural-networks) . Thanks to the contributors of [object-detection-with-spiking-neural-networks](https://github.com/loiccordone/object-detection-with-spiking-neural-networks) .

```
@InProceedings{Cordone_2022_IJCNN,
    author    = {Cordone, Loic and Miramond, Benoît and Thierion, Phillipe},
    title     = {Object Detection with Spiking Neural Networks on Automotive Event Data},
    booktitle = {Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN)},
    month     = {July},
    year      = {2022},
    pages     = {}
}
```


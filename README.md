# RandWireNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-randomly-wired-neural-networks-for/image-classification-imagenet-image-reco)](https://paperswithcode.com/sota/image-classification-imagenet-image-reco?p=exploring-randomly-wired-neural-networks-for)

![Datasets](https://img.shields.io/badge/Dataset-MNIST-lightgray.svg) ![Datasets](https://img.shields.io/badge/Dataset-CIFAR--10,100-green.svg) ![Datasets](https://img.shields.io/badge/Dataset-ImageNet--12-yellow.svg)

## Results

In small regime, C=78, WS(4,0.75)

| datasets    | top-1 acc. | top-5 acc. | epoch |
| ----------- | ---------- | ---------- | ----- |
| MNIST       | 99.60      | 100.       | 100   |
| CIFAR-10    | 91.71      | 99.75      | 250   |
| CIFAR-100   | 72.49      | 92.15      | 250   | 
| ImageNet-12 | 56.70      | 78.95      | 81    |

In regular regime, C=109, WS(4,0.75)

| datasets    | top-1 acc.       | top-5 acc. | epoch |
| ----------- | ---------------- | ---------- | ----- |
| ImageNet-12 | work in progress |            |       |

*- Because my computer does not have powerful computing power, it will take some time to update.*

## Running the example

### Setup
Please prepare:
[![Python Version](https://img.shields.io/badge/python-3.6,3.7-green.svg)](https://www.python.org/downloads/release/python-360/) [![Pytorch Version](https://img.shields.io/badge/pytorch-1.0,1.1-orange.svg)](https://pytorch.org/get-started/locally/)

Clone the repository and install the following additional packages:

```
git clone https://github.com/hebo1221/RandWireNN.git
pip install -r requirements.txt
```

### Running the demo
Just
```
python run_RandWireNN.py
```
- If you want to change dataset, see run_RandWireNN.py, get_configuration(). MNIST,CIFAR,ImageNet available
- You don't have to prepare a dataset. The code will automatically download it.
- But if you have it already, set your dataset directory in RandWireNN_config.py, __C.DATASET_DIR
- If you want to see a train-loss graph, see RandWireNN_config.py, __C.VISDOM
- You can change the hyperparameters and dataset settings from *_config.py files. Look it up.


### Reference

All details regarding the Randomly Wired Neural Networks can be found in the original research paper: [https://arxiv.org/pdf/1904.01569v2](https://arxiv.org/pdf/1904.01569v2).


## License

All materials in this repository are released under the Apache License 2.0.
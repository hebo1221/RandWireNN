# RandWireNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-randomly-wired-neural-networks-for/image-classification-imagenet-image-reco)](https://paperswithcode.com/sota/image-classification-imagenet-image-reco?p=exploring-randomly-wired-neural-networks-for) 
![Datasets](https://img.shields.io/badge/Dataset-MNIST-lightgray.svg)![Datasets](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)![Datasets](https://img.shields.io/badge/Dataset-CIFAR--100-green.svg)![Datasets](https://img.shields.io/badge/Dataset-ImageNet-12--100-yellow.svg)

I designed extra small regime for smaller size datasets(mnist,cifar-10,100).

## Results
in small regime, C=78, WS(4,0.75)
| datasets | top-1 acc. | top-5 acc. | epoch |
| :---: | :---: | :---: | :---: |
MNIST | 97.810 | 99.880 | 1
CIFAR-10 | 55.140 | 94.810 | 1
CIFAR-100 | 16.310 | 42.920 | 1
IMAGENET-12 | 56.704 | 78.953 | 81
  - Because my computer does not have powerful computing power, it will take some time to update.


## Running the example

### Setup

[![Python Version](https://img.shields.io/badge/python-3.7-green.svg)](https://www.python.org/downloads/release/python-360/) [![Pytorch Version](https://img.shields.io/badge/pytorch-1.1-orange.svg)](https://pytorch.org/get-started/locally/)

Clone the repository and install the following additional packages:

```
git clone https://github.com/hebo1221/RandWireNN.git
pip install -r requirements.txt
```

### Running the demo

To train and evaluate run

`python run_RandWireNN.py`

If you want to change dataset, see run_RandWireNN.get_configuration()
You don't have to prepare a dataset. The code will automatically download.
You can change the hyperparameters from _config.py


### Algorithm 

All details regarding the Randomly Wired Neural Networks can be found in the original research paper: [https://arxiv.org/pdf/1904.01569v2](https://arxiv.org/pdf/1904.01569v2).



## License

All materials in this repository are released under the Apache License 2.0.
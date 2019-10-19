paper8408
========================================================================

Pytorch Implementation for the paper:

**[DMRM: A Dual-channel Multi-hop Reasoning Model for Visual Dialog]**

<!--![Overview of Dual-channel Multi-hop Reasoning Model](dmrm_overview.jpg)-->
<img src="dmrm_overview.png" width="90%" align="middle">


Setup and Dependencies
----------------------
This starter code is implemented using **PyTorch v0.3.1** with **CUDA 8 and CuDNN 7**. <br>
It is recommended to set up this source code using Anaconda or Miniconda. <br>

1. Install Anaconda or Miniconda distribution based on **Python 3.6+** from their [downloads' site][2].
2. Clone this repository and create an environment:

```sh
git clone https://github.com/paper-coder/paper8408.git
conda create -n dan_visdial python=3.6

# activate the environment and install all dependencies
conda activate dmrm_visdial
cd paper8408/
pip install -r requirements.txt
```

Download Features
----------------------
1. We used the Faster-RCNN pre-trained with Visual Genome as image features. Download the image features below, and put each feature under `$PROJECT_ROOT/data` directory. 
  * [`features_faster_rcnn_x101_train.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5): Bottom-up features of 36 proposals from images of `train` split.
  * [`features_faster_rcnn_x101_val.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5): Bottom-up features of 36 proposals from images of `val` split.
  * [`features_faster_rcnn_x101_test.h5`](https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5): Bottom-up features of 36 proposals from images of `test` split.

2. Download the GloVe pretrained word vectors from [here][9], and keep `glove.6B.300d.txt` under `$PROJECT_ROOT/data` directory.

Data preprocessing & Word embedding initialization
----------------------
```sh
# data preprocessing
cd dmrm-VisDial/script/
python prepro.py

# Word embedding vector initialization (GloVe)
cd dmrm-VisDial/script/
python create_glove.py
```

Training
--------
Simple run
```sh
python train.py 
```

### Saving model checkpoints  
By default, our model save model checkpoints at every epoch. You can change it by using `-save_step` option. 

### Logging
Logging data `checkpoints/start/time/log.txt` shows epoch, loss, and learning rate.

Evaluation
--------
Evaluation of a trained model checkpoint can be evaluated as follows:
```sh
python evaluate.py -load_path /path/to/.pth -split val
```
Validation scores can be checked in offline setting. But if you want to check the `test split` score, you have to submit a json file to [online evaluation server][10]. You can make json format with `-save_ranks=True` option.

Results
--------
Performance on `v1.0 test-std` (trained on `v1.0` train):

  Model  |  NDCG   |  MRR   |  R@1  | R@5  |  R@10   |  Mean  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ |
DAN | 0.5759 | 0.6320 | 49.63 |  79.75| 89.35 | 4.30 |

License
--------
MIT License

[1]: https://arxiv.org/abs/1902.09368
[2]: https://conda.io/docs/user-guide/install/download.html
[3]: https://drive.google.com/file/d/1NYlSSikwEAqpJDsNGqOxgc0ZOkpQtom9/view?usp=sharing
[4]: https://drive.google.com/file/d/1QSi0Lr4XKdQ2LdoS1taS6P9IBVAKRntF/view?usp=sharing
[5]: https://drive.google.com/file/d/1NI5TNKKhqm6ggpB2CK4k8yKiYQE3efW6/view?usp=sharing
[6]: https://drive.google.com/file/d/1nTBaLziRIVkKAqFtQ-YIbXew2tYMUOSZ/view?usp=sharing
[7]: https://drive.google.com/file/d/1BXWPV3k-HxlTw_k3-kTV6JhWrdzXsT7W/view?usp=sharing
[8]: https://drive.google.com/file/d/1_32kGhd6wKzQLqfmqJzIHubfZwe9nhFy/view?usp=sharing
[9]: http://nlp.stanford.edu/data/glove.6B.zip 
[10]: https://evalai.cloudcv.org/web/challenges/challenge-page/161/overview

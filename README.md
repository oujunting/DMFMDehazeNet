# DMFMDehazeNet: A dehazing network for multi-dimensional attention and multi-model deep fusion

> **Abstract:** Single image dehazing has always been an important challenge in the field of computer vision due to the complexity involved in information loss and uneven noise distribution. Existing dehazing techniques usually rely on the output of a single control layer. Although this approach simplifies model design to a certain extent, it also increases the complexity of training and significantly increases the demand for computing resources. To deal with this problem, this paper proposes a dehazing method based on multi-model deep fusion, with soft light mode as the core component. Through comprehensive processing of multi-dimensional features, this method inputs the generated control layer and foggy image into the soft light mode and the other four control models to respectively enhance the light and dark contrast, detail expression, brightness and contrast of the image. Finally, Fusion outputs high-quality haze-free images. In addition, this article also developed a multi-dimensional attention module (MDA), which combines the coordinate attention mechanism with a stronger channel attention mechanism, focusing on solving the non-uniformity of fog distribution and the removal of deep haze, thereby improving Model accuracy and robustness. 
> Experimental results on the SOTS indoor data set show that the proposed method reaches 43.77dB in peak signal-to-noise ratio (PSNR), which is significantly better than the existing technology, further verifying the effectiveness and excellence of the method.

## Network Architecture
![image](https://github.com/oujunting/DMFMDehazeNet/blob/main/frame.png)


## Getting started

### Install

We test the code on PyTorch 1.10 +  CUDA 11.3

1. Create a new conda environment
 ```
conda create -n DMFMDehazeNet python=3.8
conda activate DMFMDehazeNet
```

2. Install dependencies
```
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e .
pip install -r requirements.txt
```

### Download
Since my code references [Dehazeformer](https://github.com/IDKiro/DehazeFormer#vision-transformers-for-single-image-dehazing), the dataset format is the same as that in Dehazeformer. You can choose to download the dataset from [Dehazeformer](https://github.com/IDKiro/DehazeFormer#vision-transformers-for-single-image-dehazing)

Haze4K dataset is available at [BaiduPan](https://pan.baidu.com/s/141MW0YAvjFcydlroQZZizA)(pw：cmmr）
You can download the pretrained models on [Google Drive](https://drive.google.com/drive/folders/1-ytS_ebiDM6kOjcon7rIbKtKFtb9uZvV?usp=drive_link)


The final file path will be arranged as (please check it carefully)：
```
┬─ work_dirs
│   └─ DMFMDehazeNet
│      ├─ iter_10000.pth
│      └─ ... (model name)
│   
└─ data
    ├─ RESIDE-IN
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (corresponds to the former)
    │   └─ test
    │       ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (corresponds to the former)
    └─ ... (dataset name)
```

### Model Training
You can download our model from [Google Drive](https://drive.google.com/drive/folders/1-ytS_ebiDM6kOjcon7rIbKtKFtb9uZvV?usp=drive_link)


Train the model on the `ITS` dataset using four cards
```
cd  DMFMDehazeNet 
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh ./my_config/DMFMDehazeNet.py 4
```

Generate images using pre-trained models
```
cd DMFMDehazeNet
python ./my_config/generation_demo.py --checkpoint your_path/iter_10000.pth --path your_path/reside-indoor
```


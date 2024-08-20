

### Model Training
You can download our model from [Google Drive](https://drive.google.com/drive/folders/1-ytS_ebiDM6kOjcon7rIbKtKFtb9uZvV?usp=drive_link)


Train the model on the `ITS` dataset using four cards
```
cd my_config
 CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh ./my_config/MixDehazeNet_all.py 4
```

Generate images using pre-trained models
```
cd my_config
python generation_demo.py --checkpoint your_path/iter_10000.pth --path your_path/reside-indoor
```
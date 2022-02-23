# Future_Frame_Prediction


### The network pipeline.  
TBA

## Environments  
TBA 

## Prepare
- Download the ped2 and avenue datasets.  

|USCD Ped2                                                                           | CUHK Avenue                                                                        | CalTech Pedestrian Dataset                          |
|:----------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:| :--------------------------------------------------:|
|[Google Drive](https://drive.google.com/open?id=1PO5BCMHUnmyb4NRSBFu28squcDv5VWTR)  | [Google Drive](https://drive.google.com/open?id=1b1q0kuc88rD5qDf5FksMwcJ43js4opbe) | [Google Drive](https://drive.google.com/drive/folders/1IBlcJP8YsCaT81LwQ2YwQJac8bf1q8xF) |



## Train Generator
```Shell
# Train by default with specified dataset.
python train_gen.py --dataset=avenue
# Train with different batch_size, you might need to tune the learning rate by yourself.
python train_gen.py --dataset=avenue --batch_size=16
# Set the max training iterations.
python train_gen.py --dataset=avenue --iters=80000
# Set the save interval and the validation interval.
python train_gen.py --dataset=avenue --save_interval=2000 --val_interval=2000
# Resume training with the latest trained model or a specified model.
python train_gen.py --dataset=avenue --resume latest [or avenue_10000.pth]

```

## Finetuning
```Shell
# Finetuning by default with specified dataset.
python ft_dqn.py --dataset=CalTech --resume_g=CalTech_10000.pth
# Finetuning with different batch_size, you might need to tune the learning rate by yourself.
python ft_dqn.py --dataset=CalTech --resume_g=CalTech_10000.pth --batch_size=16
# Set the max training iterations.
python ft_dqn.py --dataset=CalTech --resume_g=CalTech_10000.pth --iters=80000
# Set the save interval and the validation interval.
python ft_dqn.py --dataset=CalTech --resume_g=CalTech_10000.pth --save_interval=2000 --val_interval=2000
# Resume training with the trained RL model.
python ft_dqn.py --dataset=CalTech --resume_g=CalTech_10000.pth --resume_r=ft_CalTech_10000.pth

```

## Use tensorboard
```Shell
tensorboard --logdir=tensorboard_log/ped2_bs4
```

## Evalution
```Shell
# Validate with a trained model.
python evaluate_ft.py --dataset=CalTech --trained_model=CalTech_10000.pth
```

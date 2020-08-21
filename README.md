# PANDA Challenge

My 120th place solution and writeup to the [PANDA Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment) hosted on Kaggle by Radboud University Medical Center and Karolinska Institute.

![](https://github.com/GreatGameDota/PANDA-Challenge-Solution/blob/master/assets/main.png)

## Initial Thoughts

I will keep this overview short and exclude any long winded stories and detailed explanations. Never-the-less this was another great Kaggle competition I was glad to partipate in!

## Overview

My best solution on the private leaderboard was an ensemble and TTA of 3 models: 1 EfficientNet B0 and 2 EfficientNet B1s. There were all trained on Google Colab Pro for 10-15 epochs. I used the same tiling technique as [@iafoss](https://www.kaggle.com/iafoss) shared early on in the competition.

## Model

Simple Pretrained Model w/ GeM pooling -> Flatten -> Dropout(0.5) -> Linear

All models used pretrained weights from [PytorchCV](https://github.com/osmr/imgclsmob/tree/master/pytorch) and use the same label binning technique shared by [@haqishen](https://www.kaggle.com/haqishen) with BCE loss.

I also started to use [Nvidia's Apex](https://github.com/NVIDIA/apex) for mixed precision training.

## Input and Augmentation

As input I used no External Data and used the tilting method shared by @iafoss which can be found [here](https://www.kaggle.com/iafoss/panda-16x128x128-tiles). I split each slide into 36 tiles which are 256px by 256px.

For augmentation I applied it to both each individual tile and to the entire image. For each tile I applied SSR and affine augmentations alone with flipping and coarse dropout. For the entire stiched together image I applied RandomBrightnessContrast and same flipping.

## Training

Training was simple: for B0 I trained for 10 epochs and B1 trained for 15 epochs. Both had a batch size of 4 with CosineAnnealingLR scheduler and Adam optimizer.

I used @haqishen's label binning method with BCE loss which can be found [here](https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87).

## Ensembling and TTA

I used simple ensemble + TTA to blend multiple models which boosted my public leaderboard score. More detail and my implementation can be found in the PANDA Inference TTA notebook in this repo.

## Final Submission

For my final submission I choose my two best scoring public LB submissions. Unfortunatley they were not my highest scoring submissions so I shook down quite a bit.

## What didn't work

- Any other models besides B0 and B1
- Cross entropy loss with 6 classes
- Batch Accumulation
- Regression
- More epochs
- RAdam/AdamW
- Warmup Schedulers Linear/Cosine

## Final Thoughts

Finally I worked on this competition for around 4 months and it was a lot of fun! Its unfortunate I wasn't able to survive shake up and obtain another medal but there is always next time!

My previous competition: [Deepfake Detection](https://github.com/GreatGameDota/Deepfake-Detection)

My next competition: [Melanoma Classification](https://github.com/GreatGameDota/SIIM-ISIC-Melanoma-Classification)

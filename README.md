# Fast and Easy to use video feature extractor

This repo aims at providing an easy to use and efficient code for extracting
video features using deep CNN (2D or 3D).

It has been originally designed to extract video features for the large scale video dataset HowTo100M (https://www.di.ens.fr/willow/research/howto100m/) in an efficient manner.


Most of the time, extracting CNN features from video is cumbersome.
In fact, this usually requires dumping video frames into the disk, loading the dumped frames one
by one, pre processing them and use a CNN to extract features on chunks of videos.
This process is not efficient because of the dumping of frames on disk which is
slow and can use a lot of inodes when working with large dataset of videos.

To avoid having to do that, this repo provides a simple python script for that task: Just provide a list of raw videos and the script will take care of on the fly video decoding (with ffmpeg) and feature extraction using state-of-the-art models. While being fast, it also happen to be very convenient.

This script is also optimized for multi processing GPU feature extraction.


# Requirements
- Python 3
- PyTorch (>= 1.0)
- ffmpeg-python (https://github.com/kkroening/ffmpeg-python)

# How To Use ?

First of all you need to generate a csv containing the list of videos you
want to process. For instance, if you have video1.mp4 and video2.webm to process,
you will need to generate a csv of this form:

```
video_path,feature_path
absolute_path_video1.mp4,absolute_path_of_video1_features.npy
absolute_path_video2.webm,absolute_path_of_video2_features.npy
```

And then just simply run:

```sh
python extract.py --csv=input.csv --type=2d --batch_size=64 --num_decoding_thread=4
```
This command will extract 2d video feature for video1.mp4 (resp. video2.webm) at path_of_video1_features.npy (resp. path_of_video2_features.npy) in
a form of a numpy array.
To get feature from the 3d model instead, just change type argument 2d per 3d.
The parameter --num_decoding_thread will set how many parallel cpu thread are used for the decoding of the videos.

Please note that the script is intended to be run on ONE single GPU only.
if multiple gpu are available, please make sure that only one free GPU is set visible
by the script with the CUDA_VISIBLE_DEVICES variable environnement for example.

# Can I use multiple GPU to speed up feature extraction ?

Yes ! just run the same script with same input csv on another GPU (that can be from a different machine, provided that the disk to output the features is shared between the machines). The script will create a new feature extraction process that will only focus on processing the videos that have not been processed yet, without overlapping with the other extraction process already running.

# What models are implemented ?
So far, only one 2D and one 3D models can be used.

- The 2D model is the pytorch model zoo ResNet-152 pretrained on ImageNet. The 2D features are extracted at 1 feature per second at the resolution of 224.
- The 3D model is a ResNexT-101 16 frames (https://github.com/kenshohara/3D-ResNets-PyTorch) pretrained on Kinetics. The 3D features are extracted at 1.5 feature per second at the resolution of 112.

# Downloading pretrained models
This will download the pretrained 3D ResNext-101 model we used from: https://github.com/kenshohara/3D-ResNets-PyTorch 

```sh
mkdir model
cd model
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/resnext101.pth
```



# Acknowledgements
The code re-used code from https://github.com/kenshohara/3D-ResNets-PyTorch
for 3D CNN.

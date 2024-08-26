# MultiChannel_RIR_Generation

Source code for "Development of a Microphone Array Room Impulse Response Dataset for Evaluating Multichannel Acoustic Generation (多チャンネル音響生成を評価するためのマイクロフォンアレイ室内インパルス応答データセットの構築)".  
This source code is based on [Learning Neural Acoustic Fields](https://github.com/aluo-x/Learning_Neural_Acoustic_Fields).

## Repository detail
Neural acoustic fields learn from a Room Impulse Response (RIR) dataset and can estimate RIR at arbitrary positions. We extended Neural Acoustic Fields to be able to learn from multichannel RIRs.  
Additionally, we made it possible to construct datasets through Pyroomacoustics simulations and evaluate multichannel RIR estimation based on the accuracy of Direction of Arrival (DoA), i.e., sound source direction estimation.

## Requirements (in addition to the usual python stack)
- Pytorch 1.9 (1.10 should work as well)
- Pyroomacoustics 0.7.3
- h5py
- numpy
- scipy
- matplotlib
- sklearn (for linear probe and feature visualization)
- librosa (for training data parsing)
- ffmpeg 5.0 (for AAC-LC baseline only) - compile/use docker
- opus-tools 0.2 & libopus 1.3.1 (for Xiph-opus baseline only) - install opus-tools via conda-forge
- Tested on Ubuntu 20.04 and 21.10
 

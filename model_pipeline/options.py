import argparse
import torch
import os
import random
import numpy as np

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def list_float_flag(s):
    return [float(_) for _ in list(s)]

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        parser = self.parser
        parser.add_argument('--save_loc', default="./results", type=str)


        parser.add_argument('--exp_name', default="{}")

        # dataset arguments
        parser.add_argument('--coor_base', default="./wav_data", type=str)  # Location of the training index to coordinate mapping
        parser.add_argument('--spec_base', default="./magnitudes", type=str)  # Location of the actual training spectrograms
        parser.add_argument('--phase_base', default="./phases", type=str)  # Location of the actual training spectrograms
        parser.add_argument('--mean_std_base', default="./magnitude_mean_std", type=str)  # Location of sound mean_std data
        parser.add_argument('--phase_std_base', default="./phase_std", type=str)  # Location of sound phase_std data
        parser.add_argument('--minmax_base', default="./minmax", type=str)  # Location of room bbox data
        parser.add_argument('--wav_base', default="./wav_data/raw", type=str)  # Location of impulses in raw .wav format
        parser.add_argument('--split_loc', default="./train_test_split/", type=str) # Where the train test split is stored


        # baseline arguments
        parser.add_argument('--opus_enc', default="./bin/opusenc", type=str)  # Opus encoder path -- opusenc opus-tools 0.2 (using libopus 1.3.1)
        parser.add_argument('--opus_dec', default="./bin/opusdec", type=str)  # Opus decoder path -- opusdec opus-tools 0.2 (using libopus 1.3.1)
        parser.add_argument('--ffmpeg', default="./ffmpeg-5.0-amd64-static/ffmpeg", type=str)  # ffmpeg 5.0 path, used for AAC-LC -- ffmpeg version 5.0-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2022 the FFmpeg developers built with gcc 8 (Debian 8.3.0-6)
        parser.add_argument('--aac_write_path', default="./aac_enc_test", type=str) # Where do we write the aac encoded-decoded data
        parser.add_argument('--opus_write_path', default="./opus_enc_test", type=str) # Where do we write the opus encoded-decoded data
        parser.add_argument('--ramdisk_path', default="/mnt/ramdisk", type=str) # RAMdisk for acceleration

        # training arguments
        parser.add_argument('--gpus', default=4, type=int) # Number of GPUs to use
        parser.add_argument('--epochs', default=200, type=int) # Total epochs to train for
        parser.add_argument('--resume', default=0, type=bool_flag) # Load weights or not from latest checkpoint
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--reg_eps', default=1e-1, type=float) # Noise to regularize positions
        parser.add_argument('--pixel_count', default=2000, type=int)  # Noise to regularize positions
        parser.add_argument('--lr_init', default=5e-4, type=float)  # Starting learning rate
        parser.add_argument('--lr_decay', default=1e-1, type=float)  # Learning rate decay rate
        parser.add_argument('--phase_alpha', default=1e-1, type=float)  # Phase learning scale
        parser.add_argument('--mag_alpha', default=1.0, type=float)  # Magnitude learning scale

        # network arguments
        parser.add_argument('--layers', default=8, type=int) # Number of layers in the network
        parser.add_argument('--layers_residual', default=0, type=int) # Number of residual layers in the network
        parser.add_argument('--batch_norm', default="none", type=str)
        parser.add_argument('--activation_func_name', default="default", type=str)
        parser.add_argument('--grid_gap', default=0.25, type=float) # How far are the grid points spaced
        parser.add_argument('--bandwith_init', default=0.25, type=float) # Initial bandwidth of the grid
        parser.add_argument('--features', default=512, type=int) # Number of neurons in the network for each layer
        parser.add_argument('--grid_features', default=64, type=int) # Number of neurons in the grid
        parser.add_argument('--position_float', default=0.1, type=float) # Amount the position of each grid cell can float (up or down)
        parser.add_argument('--min_bandwidth', default=0.1, type=float) # Minimum bandwidth for clipping
        parser.add_argument('--max_bandwidth', default=0.5, type=float) # Maximum bandwidth for clipping
        parser.add_argument('--num_freqs', default=10, type=int) # Number of frequency for sin/cos
        parser.add_argument('--dir_ch', default=4, type=int)  # num of direction channel(if binaural, 2)
        parser.add_argument('--max_len', default=245, type=int)  # max legth of time

        # testing arguments
        parser.add_argument('--inference_loc', default="inference_out", type=str) # os.path.join(save_loc, inference_loc), where to cache inference results
        parser.add_argument('--gt_has_phase', default=0, type=bool_flag)  # image2reverb does not use gt phase for their GT when computing T60 error, and instead use random phase. If we use GT waveform (instead of randomizing the phase, we get lower T60 error)
        parser.add_argument('--baseline_mode', default="opus", type=str)  # Are we testing aac or opus? For baselines
        parser.add_argument('--interp_mode', default="nearest", choices=['linear', 'nearest'], type=str)  # interpolation mode. For baselines
        parser.add_argument('--fp16_interp', default=0, type=str)  # Use fp16 to save memory, essentially no change in results
        parser.add_argument('--test_checkpoint', default="latest", type=str)  # checkpoint (weight) file for testing

        # visualization arguments
        parser.add_argument('--vis_ori', default=0, type=str)  # Choose an orientation to visualize, can be 0,1,2,3; corresponding to 0,90,180,270 degrees
        parser.add_argument('--room_grid_loc', default="room_grid_coors", type=str)  # where are the points for the room stored
        parser.add_argument('--room_feat_loc', default="room_feat_coors", type=str)  # where are the points for the room stored
        parser.add_argument('--room_grid_depth', default="room_depth_grid", type=str)  # room structure
        parser.add_argument('--room_scatter_depth', default="room_depth_scatter", type=str)  # room structure
        parser.add_argument('--vis_save_loc', default="loudness_img", type=str)
        parser.add_argument('--vis_feat_save_loc', default="feat_img", type=str)
        parser.add_argument('--depth_img_loc', default="depth_img", type=str)
        parser.add_argument('--net_feat_loc', default="network_feats_scatter", type=str) # a subset of the points in this folder is used for TSNE & linear fit
        parser.add_argument('--net_feat_loc2', default="network_feats_grid", type=str)
        parser.add_argument('--emitter_loc', default=[0.5, -3.0], type=list_float_flag)  # Where do we position the emitter? [0.5, -3.0] for apartment_1, [-0.2, 0.0] for apartment_2

    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        torch.manual_seed(0)
        # random.seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print()
        return self.opt
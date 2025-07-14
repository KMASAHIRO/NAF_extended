import torchaudio
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram
import librosa
#from skimage.transform import rescale, resize
from scipy.interpolate import interp1d
import os

import gc
import pickle

# Misc functions to make spectrograms

def load_audio(path_name, use_torch=True, resample=True, resample_rate=22050):
    # returns in shape (ch, num_sample), as float32 (on Linux at least)
    # by default torchaudio is wav_arr, sample_rate
    # by default wavfile is sample_rate, wav_arr
    if use_torch:
        loaded = torchaudio.load(path_name)
        wave_data_loaded = loaded[0].numpy()
        sr_loaded = loaded[1]
    else:
        loaded = wavfile.read(path_name)
        wave_data_loaded = np.clip(loaded[1], -1.0, 1.0).T
        sr_loaded = loaded[0]

    if resample:
        if wave_data_loaded.shape[1]==0:
            print("len 0")
            assert False
        #if wave_data_loaded.shape[1]<int(sr_loaded*0.1):
        #    padded_wav = librosa.util.fix_length(wave_data_loaded, size=int(sr_loaded*0.1))
        #    resampled_wave = librosa.resample(padded_wav, orig_sr=sr_loaded, target_sr=resample_rate)
        #else:
        #    resampled_wave = librosa.resample(wave_data_loaded, orig_sr=sr_loaded, target_sr=resample_rate)
        resampled_wave = librosa.resample(wave_data_loaded, orig_sr=sr_loaded, target_sr=resample_rate)
    else:
        resampled_wave = wave_data_loaded
    return np.clip(resampled_wave, -1.0, 1.0)

def if_compute(arg):
    unwrapped_angle = np.unwrap(arg).astype(np.single)
    return np.concatenate([unwrapped_angle[:,:,0:1], np.diff(unwrapped_angle, n=1)], axis=-1)

class get_spec():
    def __init__(self, use_torch=False, power_mod=2, fft_size=512):
        self.n_fft=fft_size
        self.hop = self.n_fft//4
        if use_torch:
            assert False
            self.use_torch = True
            self.spec_transform = Spectrogram(power=None, n_fft=self.n_fft, hop_length=self.hop)
        else:
            self.power = power_mod
            self.use_torch = False
            self.spec_transform = None
        
    def transform(self, wav_data_prepad):
        wav_data = librosa.util.fix_length(wav_data_prepad, size=wav_data_prepad.shape[-1]+self.n_fft//2)
        #if wav_data.shape[1]<4410:
        #    wav_data = librosa.util.fix_length(wav_data, size=4410)
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:
            
            transformed_data = np.array(librosa.stft(wav_data,n_fft=self.n_fft, hop_length=self.hop))[:,:-1]
#         print(np.array([librosa.stft(wav_data[0],n_fft=self.n_fft, hop_length=self.hop),
#                librosa.stft(wav_data[1],n_fft=self.n_fft, hop_length=self.hop)]).shape, "OLD SHAPE")

        real_component = np.abs(transformed_data)
        img_component = np.angle(transformed_data)
        gen_if = if_compute(img_component)/np.pi
        return np.log(real_component+1e-3), gen_if, img_component


# Loop through to audio
# 1. Resample to 22050 Hz
# 2. Make each log magnitude

base_dir = "/home/ach17616qc/Pyroomacoustics/outputs/real_env_avr_16kHz_centered_NAF"
raw_path = os.path.join(base_dir, "raw")
preprocess_base_dir = os.path.join(base_dir, "preprocess")
mag_path = os.path.join(preprocess_base_dir, "magnitudes")
phase_path = os.path.join(preprocess_base_dir, "phases")
spec_getter = get_spec()

# Create the directory if the path doesn't exist
path_list = [mag_path, phase_path]
for path in path_list:
    # If the path is a file, extract the directory part
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

length_tracker = []
mag_object = os.path.join(mag_path, "magnitudes")
phase_object = os.path.join(phase_path, "phases")
f_mag = h5py.File(mag_object+".h5", 'w')
f_phase = h5py.File(phase_object+".h5", 'w')
zz = 0

def return_lastnumber(strings):
    return int(strings.replace(".wav", "").split("_")[-1])

files = os.listdir(raw_path)
files = [_ for _ in files if "wav" in _]
files = sorted(files, key=return_lastnumber)
files_channel = dict()
for ff in files:
    tmp = ff.split(".")[0].split("_")
    pos_id = "_".join(tmp[:2])
    if pos_id in files_channel.keys():
        files_channel[pos_id].append(ff)
    else:
        files_channel[pos_id] = [ff]

print("Found {} files".format(str(len(files))))

for key in files_channel:
    zz+= 1 
    if zz % 500==0:
        print(zz)
    
    loaded_wav = list()
    channel_max_len = 0
    for ff in files_channel[key]:
        cur_file = os.path.join(raw_path, ff)
        try:
            loaded_wav_tmp = load_audio(cur_file, use_torch=True, resample_rate=16000)
            if loaded_wav_tmp.shape[-1] > channel_max_len:
                channel_max_len = loaded_wav_tmp.shape[-1]
            loaded_wav.append(loaded_wav_tmp)
        except Exception as e:
            print("0 length wav", cur_file, e)
            continue
    
    # 結合するためにパディング
    padded_wav = list()
    for wav in loaded_wav:
        padded_wav.append(librosa.util.fix_length(wav, size=channel_max_len))
    padded_wav = np.concatenate(padded_wav, axis=0)
    real_spec, img_spec, raw_phase = spec_getter.transform(padded_wav)
    length_tracker.append(real_spec.shape[2])
    f_mag.create_dataset('{}'.format(key), data=real_spec.astype(np.half))
    f_phase.create_dataset('{}'.format(key), data=img_spec.astype(np.half))
print("Max length", np.max(length_tracker))
max_len = np.max(length_tracker)
f_mag.close()
f_phase.close()

# Compute mean std

def pad(input_arr, max_len_in, constant=np.log(1e-3)):
    return np.pad(input_arr, [[0,0],[0,0],[0,max_len_in-input_arr.shape[2]]], constant_values=constant)

raw_path = os.path.join(preprocess_base_dir, "magnitudes")
mean_std = os.path.join(preprocess_base_dir, "magnitude_mean_std")

# Create the directory if the path doesn't exist
path_list = [mean_std]
for path in path_list:
    # If the path is a file, extract the directory part
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

files = os.listdir(raw_path)
f_name = "magnitudes.h5"
print("Processing ", f_name)
f = h5py.File(os.path.join(raw_path, f_name), 'r')
keys = list(f.keys())
all_arrs = []
for idx in np.random.choice(len(keys), size=len(keys), replace=False):  
    all_arrs.append(pad(f[keys[idx]], max_len).astype(np.single))
all_arrs_2 = np.array(all_arrs, dtype=np.single)
print("Computing mean")
mean_val = np.mean(all_arrs, axis=(0,1))
print("Computing std")
std_val = np.std(all_arrs, axis=(0,1))+0.1

print(mean_val.shape)
print("magnitude mean", np.mean(mean_val))
print("magnitude std", np.mean(std_val))
del all_arrs
f.close()
gc.collect()
with open(os.path.join(mean_std, "magnitude_mean_std.pkl"), "wb") as mean_std_file:
    pickle.dump([mean_val, std_val], mean_std_file)


# Compute phase std

def phase_pad(input_arr, max_len_in, constant=0.0):
    return np.pad(input_arr, [[0,0],[0,0],[0,max_len_in-input_arr.shape[2]]], constant_values=constant)

raw_path = os.path.join(preprocess_base_dir, "phases")
phase_std = os.path.join(preprocess_base_dir, "phase_std")

# Create the directory if the path doesn't exist
path_list = [phase_std]
for path in path_list:
    # If the path is a file, extract the directory part
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

files = os.listdir(raw_path)
f_name = "phases.h5"
print("Processing ", f_name)
f = h5py.File(os.path.join(raw_path, f_name), 'r')
keys = list(f.keys())
all_arrs = []
for idx in np.random.choice(len(keys), size=len(keys), replace=False):  
    all_arrs.append(phase_pad(f[keys[idx]], max_len).astype(np.single))
all_arrs_2 = np.array(all_arrs, dtype=np.single)
print("Computing std")
std_val = np.std(all_arrs)

print("phase std", std_val)
del all_arrs
f.close()
gc.collect()
with open(os.path.join(phase_std, "phase_std.pkl"), "wb") as phase_std_file:
    pickle.dump(std_val, phase_std_file)

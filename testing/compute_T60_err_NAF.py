from inspect import getsourcefile
import os
import numpy as np
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import get_wave, compute_t60, load_audio, get_waves
from options import Options
import h5py
import pickle
output_wav = dict()
cur_args = Options().parse()

exp_name = cur_args.exp_name

exp_dir = os.path.join(cur_args.save_loc, exp_name)
cur_args.exp_dir = exp_dir

result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
cur_args.result_output_dir = result_output_dir

save_name = os.path.join(cur_args.result_output_dir, "output_test_NAF.pkl")# +"dual_grid1"
saver_obj = h5py.File(save_name, "r")

std = saver_obj["std"][:]+0.0
mean = saver_obj["mean"][:]+0.0
phase_std = saver_obj["phase_std"]

keys = list(saver_obj.keys())
keys_new = []
for k in keys:
    if not k in ["mean", "std", "phase_std"]:
        keys_new.append(k.split("]")[0]+"]")
all_keys = list(set(keys_new))
loss = 0
total = 0
all_t60 = []
offset = 0
for k in all_keys:
    offset += 1
    if offset%1000==0:
        print(offset)
    net_out = saver_obj[k + "_out_mag"][:]
    gt_out = saver_obj[k + "_gt_mag"][:]
    actual_spec_len = net_out.shape[-1]
    std_ = std[:, :, :actual_spec_len]
    mean_ = mean[:, :, :actual_spec_len]

    net_out = (net_out * std_ + mean_)[0]
    gt_out = (gt_out * std_ + mean_)[0]

    net_phase = saver_obj[k + "_out_phase"][:][0]*phase_std
    gt_phase = saver_obj[k + "_gt_phase"][:][0]*phase_std
    
    #node_names = k.replace("[", "").replace("]", "").replace("'", "").split(",")
    #first = str(int(node_names[0]))
    #second = str(int(node_names[1]))
    #audio_file_name = os.path.join(cur_args.wav_base, "{}_{}.wav".format(first, second))
    #gt_wav2 = load_audio(audio_file_name)
    ##################
    net_wav = get_waves(net_out, net_phase, cur_args.dir_ch)
    gt_wav = get_waves(gt_out, gt_phase, cur_args.dir_ch)
    t60s = compute_t60(gt_wav, net_wav, cur_args.dir_ch)
    all_t60.append(t60s)
    output_wav[k] = {"net_wav": net_wav, "gt_wav": gt_wav}

t60s_np = np.array(all_t60)

diff = np.abs(t60s_np[:,cur_args.dir_ch:]-t60s_np[:,:cur_args.dir_ch])/np.abs(t60s_np[:,:cur_args.dir_ch])
mask = np.any(t60s_np<-0.5, axis=1)
diff = np.mean(diff, axis=1)
diff[mask] = 1

with open("./results/inference_wav/" + "output_test_NAF.pkl", mode="wb") as f:
    pickle.dump(output_wav, f)
print("{} total invalids out of {}".format(np.sum(mask), mask.shape[0]))
print(np.mean(diff)*100)
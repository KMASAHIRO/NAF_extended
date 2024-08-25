import pickle
from inspect import getsourcefile
import os

import os.path as path, sys

import numpy as np

current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from testing.test_utils import spectral
from options import Options
import h5py


cur_args = Options().parse()
exp_name = cur_args.exp_name

exp_dir = os.path.join(cur_args.save_loc, exp_name)
cur_args.exp_dir = exp_dir

result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
cur_args.result_output_dir = result_output_dir

save_name = os.path.join(cur_args.result_output_dir, "output_train_NAF.pkl")
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
phase_loss = 0
def get_stats(in_val):
    return np.mean(in_val), np.max(in_val), np.min(in_val)

offset = 0
for k in all_keys:
    offset += 1
    if offset %1000==0:
        print(offset)
    net_out = saver_obj[k+"_out_mag"][:]
    gt_out = saver_obj[k+"_gt_mag"][:]
    actual_spec_len = net_out.shape[-1]
    std_ = std[:, :, :actual_spec_len]
    mean_ = mean[:, :, :actual_spec_len]

    net_out_phase = saver_obj[k+"_out_phase"][:]*phase_std
    gt_out_phase = saver_obj[k+"_gt_phase"][:]*phase_std
    phase_loss += spectral(net_out_phase, gt_out_phase)

    net_out = (net_out*std_ + mean_)[0]
    gt_out = (gt_out * std_ + mean_)[0]
    loss += spectral(net_out, gt_out)
    total += 1.0

mean_loss = loss/total
mean_loss_phase = phase_loss/total
print("the spectral loss is {}".format(mean_loss))
print("the spectral phase loss is {}".format(mean_loss_phase))
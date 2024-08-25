import torch
torch.backends.cudnn.benchmark = True
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from data_loading.sound_loader import soundsamples
import pickle
import os
from model.networks import kernel_residual_fc_embeds
from model.modules import embedding_module_log
import math
from options import Options
import h5py

def to_torch(input_arr):
    return input_arr[None]

def test_net(rank, other_args):
    pi = math.pi
    output_device = rank
    print("creating dataset")
    dataset = soundsamples(other_args)
    xyz_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    auditory_net = kernel_residual_fc_embeds(input_ch=126, dir_ch=other_args.dir_ch, output_ch=2, intermediate_ch=other_args.features, grid_ch=other_args.grid_features, num_block=other_args.layers, num_block_residual=other_args.layers_residual, grid_gap=other_args.grid_gap, grid_bandwidth=other_args.bandwith_init, bandwidth_min=other_args.min_bandwidth, bandwidth_max=other_args.max_bandwidth, float_amt=other_args.position_float, min_xy=dataset.min_pos, max_xy=dataset.max_pos, batch_norm=other_args.batch_norm, batch_norm_features=other_args.pixel_count, activation_func_name=other_args.activation_func_name).to(output_device)

    loaded_weights = False
    current_files = sorted(os.listdir(other_args.exp_dir))
    if len(current_files)>0:
        if other_args.test_checkpoint == "latest":
            weights_file = current_files[-1]
        else:
            weights_file = other_args.test_checkpoint + ".chkpt"
        print("Identified checkpoint {}".format(weights_file))
        map_location = 'cuda:%d' % rank
        weight_loc = os.path.join(other_args.exp_dir, weights_file)
        weights = torch.load(weight_loc, map_location=map_location)
        print("Checkpoint loaded {}".format(weight_loc))
        auditory_net.load_state_dict(weights["network"])
        loaded_weights = True
    if loaded_weights is False:
        print("Weights not found")

    auditory_net.eval()
    container = dict()
    save_name = os.path.join(other_args.result_output_dir, "output_test_NAF.pkl")
    # container["mean_std"] = (dataset.std.numpy(), dataset.mean.numpy())

    saver_obj = h5py.File(save_name, "w")
    saver_obj.create_dataset("mean", data=dataset.mean.numpy())
    saver_obj.create_dataset("std", data=dataset.std.numpy())
    saver_obj.create_dataset("phase_std", data=dataset.phase_std)

    with torch.no_grad():
        num_sample_test = len(dataset.sound_files_test)
        offset = 0
        print("Total {}".format(num_sample_test))
        for test_id in range(num_sample_test):
            offset += 1
            if offset%100 == 0:
                print("Currently on {}".format(offset))
            data_stuff = dataset.get_item_test(test_id)
            gt = to_torch(data_stuff[0])
            position = to_torch(data_stuff[1]).to(output_device, non_blocking=True)
            non_norm_position = to_torch(data_stuff[2]).to(output_device, non_blocking=True)
            freqs = to_torch(data_stuff[3]).to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
            times = to_torch(data_stuff[4]).to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
            PIXEL_COUNT = other_args.pixel_count
            PIXEL_COUNT_test = gt.shape[-1]
            position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT_test, -1)
            freq_embed = freq_embedder(freqs)
            time_embed = time_embedder(times)
            total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
            output_list = list()
            for split_id in range(-(-PIXEL_COUNT_test//PIXEL_COUNT)):
                total_in_split = total_in[:, split_id*PIXEL_COUNT:(split_id+1)*PIXEL_COUNT, :]
                if total_in_split.shape[1] < PIXEL_COUNT:
                    pad_data = torch.zeros(total_in_split.shape[0], PIXEL_COUNT-total_in_split.shape[1], total_in_split.shape[2]).to(output_device, non_blocking=True)
                    total_in_split_padded = torch.cat((total_in_split, pad_data), dim=1)
                    output_split = auditory_net(total_in_split_padded, non_norm_position.squeeze(1)).transpose(1, 2)
                    output_split = output_split[:, :, :total_in_split.shape[1], :]
                else:
                    output_split = auditory_net(total_in_split, non_norm_position.squeeze(1)).transpose(1, 2)
                output_list.append(output_split)
            output = torch.cat(output_list, dim=2)
            #output = auditory_net(total_in, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
            myout = output.cpu().numpy()
            myout_mag = myout[...,0].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])
            myout_phase = myout[...,1].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])
            mygt = gt.numpy()
            mygt_mag = mygt[:,:other_args.dir_ch].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])
            mygt_phase = mygt[:,other_args.dir_ch:].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])
            name = "{}".format(dataset.sound_name)
            saver_obj.create_dataset(name+"_out_mag", data=myout_mag+0.0)
            saver_obj.create_dataset(name+"_out_phase", data=myout_phase+0.0)
            saver_obj.create_dataset(name+"_gt_mag", data=mygt_mag+0.0)
            saver_obj.create_dataset(name+"_gt_phase", data=mygt_phase+0.0)
    saver_obj.close()
    return 1


if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name

    exp_dir = os.path.join(cur_args.save_loc, exp_name)
    cur_args.exp_dir = exp_dir

    result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    cur_args.result_output_dir = result_output_dir
    if not os.path.isdir(result_output_dir):
        os.mkdir(result_output_dir)

    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, need checkpoint folder...".format(cur_args.save_loc))
        exit()
    if not os.path.isdir(cur_args.exp_dir):
        print("Experiment {} does not exist, need experiment folder...".format(cur_args.exp_name))
        exit()
    print("Experiment directory is {}".format(exp_dir))
    world_size = cur_args.gpus
    test_ = test_net(0, cur_args)

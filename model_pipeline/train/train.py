import torch
torch.backends.cudnn.benchmark = True

from model_pipeline.sound_loader import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
from model.networks import kernel_residual_fc_embeds
from model.modules import embedding_module_log
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from time import time
from model_pipeline.options import Options
import functools
import random
import pyroomacoustics as pra
import shutil
import logging
import pickle

def get_spectrograms(input_stft, input_if):
    # 8 chanel input of shape [8,freq,time]
    padded_input_stft = np.concatenate((input_stft, input_stft[:,-1:]), axis=1)
    padded_input_if = np.concatenate((input_if, input_if[:,-1:]), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1)*np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    restored = (np.exp(padded_input_stft)-1e-3)*phase_val
    return restored

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def worker_init_fn(worker_id, myrank_info):
    # print(worker_id + myrank_info*100, "SEED")
    np.random.seed(worker_id + myrank_info*100)
    random.seed(worker_id + myrank_info*100)

def train_net(rank, world_size, freeport, other_args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = freeport
    output_device = rank
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    pi = math.pi
    PIXEL_COUNT=other_args.pixel_count

    dataset = soundsamples(other_args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    ranked_worker_init = functools.partial(worker_init_fn, myrank_info=rank)
    sound_loader = torch.utils.data.DataLoader(dataset, batch_size=other_args.batch_size//world_size, shuffle=False, num_workers=3, worker_init_fn=ranked_worker_init, persistent_workers=True, sampler=train_sampler,drop_last=False)

    xyz_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)

    auditory_net = kernel_residual_fc_embeds(input_ch=126, dir_ch=other_args.dir_ch, output_ch=2, intermediate_ch=other_args.features, grid_ch=other_args.grid_features, num_block=other_args.layers, num_block_residual=other_args.layers_residual, grid_gap=other_args.grid_gap, grid_bandwidth=other_args.bandwith_init, bandwidth_min=other_args.min_bandwidth, bandwidth_max=other_args.max_bandwidth, float_amt=other_args.position_float, min_xy=dataset.min_pos, max_xy=dataset.max_pos, batch_norm=other_args.batch_norm, batch_norm_features=other_args.pixel_count, activation_func_name=other_args.activation_func_name).to(output_device)

    if rank == 0:
        print("Dataloader requires {} batches".format(len(sound_loader)))
        
        loss_logger = logging.getLogger("loss_logger")
        loss_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(other_args.exp_dir, "loss.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        loss_logger.addHandler(fh)

    start_epoch = 1
    load_opt = 0
    loaded_weights = False
    if other_args.resume:
        if not os.path.isdir(other_args.exp_dir):
            print("Missing save dir, exiting")
            dist.barrier()
            dist.destroy_process_group()
            return 1
        else:
            current_files = sorted(os.listdir(other_args.exp_dir))
            if len(current_files)>0:
                latest = current_files[-1]
                start_epoch = int(latest.split(".")[0]) + 1
                if rank == 0:
                    print("Identified checkpoint {}".format(latest))
                if start_epoch >= (other_args.epochs+1):
                    dist.barrier()
                    dist.destroy_process_group()
                    return 1
                map_location = 'cuda:%d' % rank
                weight_loc = os.path.join(other_args.exp_dir, latest)
                weights = torch.load(weight_loc, map_location=map_location)
                if rank == 0:
                    print("Checkpoint loaded {}".format(weight_loc))
                auditory_net.load_state_dict(weights["network"])
                loaded_weights = True
                if "opt" in weights:
                    load_opt = 1
                dist.barrier()
        if loaded_weights is False:
            print("Resume indicated, but no weights found!")
            dist.barrier()
            dist.destroy_process_group()
            exit()

    # We have conditional forward, must set find_unused_parameters to true
    ddp_auditory_net = DDP(auditory_net, find_unused_parameters=True, device_ids=[rank])
    criterion = torch.nn.MSELoss()
    orig_container = []
    grid_container = []
    for par_name, par_val in ddp_auditory_net.named_parameters():
        if "grid" in par_name:
            grid_container.append(par_val)
        else:
            orig_container.append(par_val)

    optimizer = torch.optim.AdamW([
        {'params': grid_container, 'lr': other_args.lr_init, 'weight_decay': 1e-2},
        {'params': orig_container, 'lr': other_args.lr_init, 'weight_decay': 0.0}], lr=other_args.lr_init, weight_decay=0.0)

    if load_opt:
        print("loading optimizer")
        optimizer.load_state_dict(weights["opt"])
        dist.barrier()

    best_doa_values = (180*np.ones(10)).tolist()
    best_doa_chkpt_list = ["" for i in range(10)]
    if rank == 0:
        old_time = time()
    for epoch in range(start_epoch, other_args.epochs+1):
        total_losses = 0
        total_mag_loss = 0
        total_phase_loss = 0
        total_spectral_loss = 0
        cur_iter = 0
        ddp_auditory_net.train()
        for data_stuff in sound_loader:
            gt = data_stuff[0].to(output_device, non_blocking=True)
            position = data_stuff[1].to(output_device, non_blocking=True)
            non_norm_position = data_stuff[2].to(output_device, non_blocking=True)
            freqs = data_stuff[3].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
            times = data_stuff[4].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi

            with torch.no_grad():
                position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
                freq_embed = freq_embedder(freqs)
                time_embed = time_embedder(times)

            total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
            optimizer.zero_grad(set_to_none=False)
            try:
                output = ddp_auditory_net(total_in, non_norm_position.squeeze(1)).transpose(1, 2)
                #output = 0
            except Exception as foward_exception:
                print(gt.shape, position.shape, freqs.shape, times.shape, position_embed.shape,
                      freq_embed.shape, time_embed.shape)
                print("Failure", foward_exception)
                continue
            # output shape torch.Size([5, other_args.dir_ch, 2000, 2])
            # gt shape torch.Size([5, other_args.dir_ch*2, 2000])
            mag_loss = criterion(output[...,0], gt[:,:other_args.dir_ch]) * other_args.mag_alpha
            phase_loss = criterion(output[...,1], gt[:,other_args.dir_ch:]) * other_args.phase_alpha
            loss = mag_loss + phase_loss
            # spectral_loss = spectral_criterion(output[...,0], gt[:,:2])
            if rank==0:
                total_mag_loss += mag_loss.detach()
                total_phase_loss += phase_loss.detach()
                # total_spectral_loss += spectral_loss.detach()
                total_losses += loss.detach()
                cur_iter += 1
            loss.backward()
            optimizer.step()
        decay_rate = other_args.lr_decay
        new_lrate_grid = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))
        new_lrate = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))

        par_idx = 0
        for param_group in optimizer.param_groups:
            if par_idx == 0:
                param_group['lr'] = new_lrate_grid
            else:
                param_group['lr'] = new_lrate
            par_idx += 1

        # ---------- 評価結果保存用 ----------
        ori_sig_spec_list = []
        pred_sig_spec_list = []
        position_rx_list = []
        position_tx_list = []
                
        total_losses_val = 0
        total_mag_loss_val = 0
        total_phase_loss_val = 0
        cur_iter_val = 0
        DoA_err = 0
        DoA_cur_iter_val = 0
        net_spec_ch_list = []
        gt_spec_ch_list = []
        ddp_auditory_net.eval()
        with torch.no_grad():
            for val_id in range(len(dataset.sound_files_val)):
                data_stuff_val = dataset.get_item_val(val_id)
                
                gt_val = data_stuff_val[0][None].to(output_device, non_blocking=True)
                position_val = data_stuff_val[1][None].to(output_device, non_blocking=True)
                non_norm_position_val = data_stuff_val[2][None].to(output_device, non_blocking=True)
                freqs_val = data_stuff_val[3][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
                times_val = data_stuff_val[4][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
        
                PIXEL_COUNT_val = gt_val.shape[-1]
                position_embed_val = xyz_embedder(position_val).expand(-1, PIXEL_COUNT_val, -1)
                freq_embed_val = freq_embedder(freqs_val)
                time_embed_val = time_embedder(times_val)

                total_in_val = torch.cat((position_embed_val, freq_embed_val, time_embed_val), dim=2)
                output_val_list = list()
                for split_id in range(-(-PIXEL_COUNT_val//PIXEL_COUNT)):
                    total_in_val_split = total_in_val[:, split_id*PIXEL_COUNT:(split_id+1)*PIXEL_COUNT, :]
                    if total_in_val_split.shape[1] < PIXEL_COUNT:
                        pad_data = torch.zeros(total_in_val_split.shape[0], PIXEL_COUNT-total_in_val_split.shape[1], total_in_val_split.shape[2]).to(output_device, non_blocking=True)
                        total_in_val_split_padded = torch.cat((total_in_val_split, pad_data), dim=1)
                        output_val_split = ddp_auditory_net(total_in_val_split_padded, non_norm_position_val.squeeze(1)).transpose(1, 2)
                        output_val_split = output_val_split[:, :, :total_in_val_split.shape[1], :]
                    else:
                        output_val_split = ddp_auditory_net(total_in_val_split, non_norm_position_val.squeeze(1)).transpose(1, 2)
        
                    mag_loss_val = criterion(output_val_split[...,0], gt_val[:,:other_args.dir_ch, split_id*PIXEL_COUNT:(split_id+1)*PIXEL_COUNT]) * other_args.mag_alpha
                    phase_loss_val = criterion(output_val_split[...,1], gt_val[:,other_args.dir_ch:, split_id*PIXEL_COUNT:(split_id+1)*PIXEL_COUNT]) * other_args.phase_alpha
                    loss_val = mag_loss_val + phase_loss_val
        
                    total_mag_loss_val += mag_loss_val
                    total_phase_loss_val += phase_loss_val
                    total_losses_val += loss_val
                    cur_iter_val += 1

                    output_val_list.append(output_val_split)
                
                output_val = torch.cat(output_val_list, dim=2)
                # Reconstruct spectrogram
                myout = output_val.cpu().numpy()
                myout_mag = myout[...,0].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])
                myout_phase = myout[...,1].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])
                mygt = gt_val.cpu().numpy()
                mygt_mag = mygt[:,:other_args.dir_ch].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])
                mygt_phase = mygt[:,other_args.dir_ch:].reshape(1, other_args.dir_ch, dataset.sound_size[1], dataset.sound_size[2])

                # padding with zero
                myout_mag = np.pad(myout_mag, ((0, 0), (0, 0), (0, 0), (0, dataset.mean.numpy().shape[-1] - dataset.sound_size[2])))
                myout_phase = np.pad(myout_phase, ((0, 0), (0, 0), (0, 0), (0, dataset.mean.numpy().shape[-1] - dataset.sound_size[2])))
                mygt_mag = np.pad(mygt_mag, ((0, 0), (0, 0), (0, 0), (0, dataset.mean.numpy().shape[-1] - dataset.sound_size[2])))
                mygt_phase = np.pad(mygt_phase, ((0, 0), (0, 0), (0, 0), (0, dataset.mean.numpy().shape[-1] - dataset.sound_size[2])))

                net_mag = (myout_mag * dataset.std.numpy() + dataset.mean.numpy())[0]
                gt_mag = (mygt_mag * dataset.std.numpy() + dataset.mean.numpy())[0]
                net_phase = myout_phase[0]*dataset.phase_std
                gt_phase = mygt_phase[0]*dataset.phase_std

                net_spec = get_spectrograms(net_mag, net_phase)
                gt_spec = get_spectrograms(gt_mag, gt_phase)

                if other_args.dir_ch <= 1:
                    net_spec_ch_list.append(net_spec)
                    gt_spec_ch_list.append(gt_spec)

                if other_args.dir_ch > 1 or len(net_spec_ch_list) == 8:
                    if other_args.dir_ch > 1:
                        doa_ch = other_args.dir_ch
                        net_spec_doa = net_spec
                        gt_spec_doa = gt_spec
                    else:
                        doa_ch = len(net_spec_ch_list)
                        net_spec_doa = np.concatenate(net_spec_ch_list, axis=0)
                        gt_spec_doa = np.concatenate(gt_spec_ch_list, axis=0)
                        net_spec_ch_list = []
                        gt_spec_ch_list = []

                    # DoA
                    position_circle_xy = pra.beamforming.circular_2D_array(center=[0,0], M=doa_ch, phi0=math.pi/2, radius=0.0365)
                    # DoA of the correct data
                    doa_gt = pra.doa.algorithms["NormMUSIC"](position_circle_xy, fs=16000, nfft=512)
                    doa_gt.locate_sources(gt_spec_doa)
                    gt_degree = np.argmax(doa_gt.grid.values)
                    # DoA of the validation data
                    doa_net = pra.doa.algorithms["NormMUSIC"](position_circle_xy, fs=16000, nfft=512)
                    doa_net.locate_sources(net_spec_doa)
                    net_degree = np.argmax(doa_net.grid.values)
                    # Calculate error
                    DoA_err += np.min([np.abs(net_degree - gt_degree), 360.0 - np.abs(net_degree - gt_degree)])
                    DoA_cur_iter_val += 1

                # --- non_norm_position_val からTx, Rx分離
                non_norm = non_norm_position_val.squeeze().cpu().numpy()
                position_tx = non_norm[:2][None]  # → shape: (1, 2)
                position_rx = non_norm[2:][None]  # → shape: (1, 2)

                # --- 蓄積（CPUに転送してnumpy化）
                pred_sig_spec_list.append(net_spec)
                ori_sig_spec_list.append(gt_spec)
                position_tx_list.append(position_tx)
                position_rx_list.append(position_rx)

        if rank == 0:
            avg_loss = total_losses.item() / cur_iter
            avg_mag = total_mag_loss.item() / cur_iter
            avg_phase = total_phase_loss.item() / cur_iter
            avg_loss_val = total_losses_val.item() / cur_iter_val
            avg_mag_val = total_mag_loss_val.item() / cur_iter_val
            avg_phase_val = total_phase_loss_val.item() / cur_iter_val
            avg_DoA_err = DoA_err / DoA_cur_iter_val
            print("{}: Ending epoch {}, loss {:.5f}, mag {:.5f}, phase {:.5f}, loss_val {:.5f}, mag_val {:.5f}, phase_val {:.5f}, DoA_err(NormMUSIC) {:.5f}, time {}".format(other_args.exp_name, epoch, avg_loss, avg_mag, avg_phase, avg_loss_val, avg_mag_val, avg_phase_val, avg_DoA_err, time() - old_time))
            loss_logger.info("{}: Epoch {}, loss {:.5f}, mag {:.5f}, phase {:.5f}, loss_val {:.5f}, mag_val {:.5f}, phase_val {:.5f}, DoA_err(NormMUSIC) {:.5f}".format(
                other_args.exp_name, epoch, avg_loss, avg_mag, avg_phase, avg_loss_val, avg_mag_val, avg_phase_val, avg_DoA_err))
            loss_save_dir = os.path.join(other_args.exp_dir, "loss_values")
            os.makedirs(loss_save_dir, exist_ok=True)
            loss_data = {
                "epoch": epoch,
                "loss": avg_loss,
                "mag": avg_mag,
                "phase": avg_phase,
                "loss_val": avg_loss_val,
                "mag_val": avg_mag_val,
                "phase_val": avg_phase_val,
                "DoA_err": avg_DoA_err
            }
            with open(os.path.join(loss_save_dir, f"loss_epoch_{epoch:05d}.pkl"), "wb") as f:
                pickle.dump(loss_data, f)
            
            old_time = time()

            save_dir = os.path.join(other_args.exp_dir, "val_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"val_epoch_{epoch:05d}.npz")
            np.savez_compressed(
                save_path,
                ori_sig_spec=np.concatenate(ori_sig_spec_list, axis=0),
                pred_sig_spec=np.concatenate(pred_sig_spec_list, axis=0),
                position_rx=np.concatenate(position_rx_list, axis=0),
                position_tx=np.concatenate(position_tx_list, axis=0),
            )
            print(f"Validation results saved to: {save_path}")
        #if rank == 0 and (epoch%20==0 or epoch==1 or epoch>(other_args.epochs-3)):
        #
        #    save_name = str(epoch).zfill(5)+".chkpt"
        #    save_dict = {}
        #    save_dict["network"] = ddp_auditory_net.module.state_dict()
        #    torch.save(save_dict, os.path.join(other_args.exp_dir, save_name))
        if rank == 0 and (epoch==1 or epoch==other_args.epochs):
            save_name = str(epoch).zfill(5)+".chkpt"
            save_dict = {}
            save_dict["network"] = ddp_auditory_net.module.state_dict()
            chkpt_dir = os.path.join(other_args.exp_dir, "chkpts")
            os.makedirs(chkpt_dir, exist_ok=True)
            save_path = os.path.join(chkpt_dir, save_name)
            torch.save(save_dict, save_path)
        
        if rank == 0 and (avg_DoA_err <= best_doa_values[-1] and avg_DoA_err < 65.0):
            replace_index = len(best_doa_values) - 1
            for i in range(len(best_doa_values)):
                if avg_DoA_err <= best_doa_values[i]:
                    replace_index = i
                    break
            best_doa_values = best_doa_values[:replace_index] + [avg_DoA_err] + best_doa_values[replace_index:-1]
            
            if best_doa_chkpt_list[replace_index] == "":
                save_name = "best_doa_" + str(replace_index+1).zfill(2) + "_epoch_" + str(epoch).zfill(5) + ".chkpt"
                best_doa_chkpt_list[replace_index] = save_name
            else:
                if best_doa_chkpt_list[-1] != "":
                    os.remove(os.path.join(other_args.exp_dir, "chkpts", best_doa_chkpt_list[-1]))
                
                for i in range(replace_index, len(best_doa_values) - 1):
                    if best_doa_chkpt_list[i] == "":
                        break
                    
                    old_save_name = best_doa_chkpt_list[i]
                    old_save_name_split = old_save_name.split("_")
                    old_save_name_split[2] = str(int(old_save_name_split[2]) + 1).zfill(2)
                    new_save_name = "_".join(old_save_name_split)
                    best_doa_chkpt_list[i] = new_save_name
                    shutil.move(os.path.join(other_args.exp_dir, "chkpts", old_save_name), os.path.join(other_args.exp_dir, "chkpts", new_save_name))
                save_name = "best_doa_" + str(replace_index+1).zfill(2) + "_epoch_" + str(epoch).zfill(5) + ".chkpt"
                best_doa_chkpt_list = best_doa_chkpt_list[:replace_index] + [save_name] + best_doa_chkpt_list[replace_index:-1]

            save_dict = {}
            save_dict["network"] = ddp_auditory_net.module.state_dict()
            chkpt_dir = os.path.join(other_args.exp_dir, "chkpts")
            os.makedirs(chkpt_dir, exist_ok=True)
            save_path = os.path.join(chkpt_dir, save_name)
            torch.save(save_dict, save_path)
            
    print("Wrapping up training {}".format(other_args.exp_name))
    dist.barrier()
    dist.destroy_process_group()
    return 1


if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name
    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, creating...".format(cur_args.save_loc))
        os.mkdir(cur_args.save_loc)
    exp_dir = os.path.join(cur_args.save_loc, exp_name)
    cur_args.exp_dir = exp_dir
    print("Experiment directory is {}".format(exp_dir))
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    world_size = cur_args.gpus
    myport = str(find_free_port())
    mp.spawn(train_net, args=(world_size, myport, cur_args), nprocs=world_size, join=True)

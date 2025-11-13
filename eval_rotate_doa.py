# eval_rotate_doa.py
# - YAMLのみで設定構成（Options().parse()は使わない）
# - total_position = [Tx_norm(2), Rx_norm(2)] / non_norm_position = [Tx(2), Rx(2)]
# - Rxを元角θ0からΔθずつ回転（θ = θ0 + k·deg_step）
# - dir_ch>1 と dir_ch==1（8本剛体回転）の両対応
# - predスペクトログラム / Tx座標 / Rx座標 / pred_deg / true_deg / used_angles / deg_step を保存
# - min/maxの範囲外は除外

import os, math, yaml, argparse
import numpy as np
import torch
import pyroomacoustics as pra
from argparse import Namespace

from model_pipeline.sound_loader import soundsamples
from model.modules import embedding_module_log
from model.networks import kernel_residual_fc_embeds
from model_pipeline.options import Options

def get_spectrograms(input_stft, input_if):
    # input: (M,F,T) の log-mag と IF
    padded_input_stft = np.concatenate((input_stft, input_stft[:, -1:]), axis=1)
    padded_input_if   = np.concatenate((input_if,   input_if[:, -1:]), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1) * np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    return (np.exp(padded_input_stft) - 1e-3) * phase_val


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def build_opt_from_yaml(yaml_path: str):
    cfg = load_yaml(os.path.expanduser(yaml_path))
    # Optionsのデフォルトを取得（CLI引数は使わない → 空でparse）
    _op = Options()
    _op.initialize()
    defaults = _op.parser.parse_args([])  # 全デフォルトが入る
    # YAMLで上書き
    for k, v in cfg.items():
        setattr(defaults, k, v)
    return defaults


def rotate_about(center_xy, radius, deg):
    th = math.radians(deg)
    cx, cy = center_xy
    return np.array([cx + radius * math.cos(th), cy + radius * math.sin(th)], dtype=float)


def in_bounds(p, min_xy, max_xy):
    return (min_xy[0] <= p[0] <= max_xy[0]) and (min_xy[1] <= p[1] <= max_xy[1])

def array_within_bounds(center_xy, radius, min_xy, max_xy):
    """円（中心 center_xy, 半径 radius）が矩形[min_xy,max_xy]に完全内包されるか"""
    x, y = center_xy
    return (
        (x - radius) >= min_xy[0]
        and (y - radius) >= min_xy[1]
        and (x + radius) <= max_xy[0]
        and (y + radius) <= max_xy[1]
    )

def angular_error_deg(a, b):
    a = np.asarray(a) % 360.0
    b = np.asarray(b) % 360.0
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


def norm_xy(xy, min_xy, max_xy):
    return np.clip(((xy - min_xy) / (max_xy - min_xy) - 0.5) * 2.0, -1.0, 1.0)


def build_model_and_embedders(opt, dataset, device):
    xyz_embedder  = embedding_module_log(num_freqs=opt.num_freqs, ch_dim=2, max_freq=7).to(device)
    time_embedder = embedding_module_log(num_freqs=opt.num_freqs, ch_dim=2).to(device)
    freq_embedder = embedding_module_log(num_freqs=opt.num_freqs, ch_dim=2).to(device)

    net = kernel_residual_fc_embeds(
        input_ch=126,
        dir_ch=opt.dir_ch,
        output_ch=2,
        intermediate_ch=opt.features,
        grid_ch=opt.grid_features,
        num_block=opt.layers,
        num_block_residual=opt.layers_residual,
        grid_gap=opt.grid_gap,
        grid_bandwidth=opt.bandwith_init,
        bandwidth_min=opt.min_bandwidth,
        bandwidth_max=opt.max_bandwidth,
        float_amt=opt.position_float,
        min_xy=dataset.min_pos,
        max_xy=dataset.max_pos,
        batch_norm=opt.batch_norm,
        batch_norm_features=opt.pixel_count,
        activation_func_name=opt.activation_func_name,
    ).to(device)
    return net, xyz_embedder, time_embedder, freq_embedder


@torch.no_grad()
def main():
    # 自前引数のみ
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     type=str, required=True)
    ap.add_argument("--ckpt",       type=str, required=True)
    ap.add_argument("--out_dir",    type=str, default=None)
    ap.add_argument("--deg_step",   type=float, default=10.0)
    ap.add_argument("--fs",         type=int,   default=16000)
    ap.add_argument("--nfft",       type=int,   default=512)
    ap.add_argument("--array_radius", type=float, default=0.0365)
    args = ap.parse_args()

    # YAMLだけで opt を作る（Options().parse() は使わない）
    opt = build_opt_from_yaml(args.config)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(opt.save_loc, opt.exp_name)
    out_dir = args.out_dir or os.path.join(exp_dir, "rotate_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Dataset
    dataset   = soundsamples(opt)
    min_xy    = np.array(dataset.min_pos, dtype=float)
    max_xy    = np.array(dataset.max_pos, dtype=float)
    mean      = dataset.mean.cpu().numpy()
    std       = dataset.std.cpu().numpy()
    phase_std = float(dataset.phase_std)

    # Model
    net, xyz_emb, time_emb, freq_emb = build_model_and_embedders(opt, dataset, device)
    state = torch.load(os.path.expanduser(args.ckpt), map_location=device)
    net.load_state_dict(state["network"])
    net.eval()

    doa_ch_conf = int(opt.dir_ch)

    # Δθ列（基準は各サンプルの θ0）
    delta_list = [k * args.deg_step for k in range(int(360 // args.deg_step))]

    summary_lines = ["unit_id,used_rotations,mean_err_deg\n"]
    all_pred, all_true = [], []

    # ===== dir_ch > 1 =====
    if doa_ch_conf > 1:
        for val_id in range(len(dataset.sound_files_val)):
            # get_item_val は total_non_norm_position = [Tx, Rx] を返す前提で扱う
            _, _, non_norm_0, freqs_norm, times_norm = dataset.get_item_val(val_id)
            nn = non_norm_0.squeeze().cpu().numpy()  # [Tx_x,Tx_y, Rx_x,Rx_y]
            tx_xy = nn[:2]
            rx_xy = nn[2:]

            # 元角θ0（Tx中心から見た Rx 方向）
            theta0 = (math.degrees(math.atan2(rx_xy[1] - tx_xy[1], rx_xy[0] - tx_xy[0])) % 360.0)
            r = float(np.linalg.norm(rx_xy - tx_xy))

            freqs = freqs_norm[None].to(device).unsqueeze(2) * (2.0 * math.pi)
            times = times_norm[None].to(device).unsqueeze(2) * (2.0 * math.pi)
            PIXEL_COUNT_val = freqs.shape[1]

            pred_deg_list, true_deg_list, used_angles = [], [], []
            rx_positions, pred_specs = [], []

            for d in delta_list:
                ang = (theta0 + d) % 360.0
                rx_rot = rotate_about(tx_xy, r, ang)
                if not array_within_bounds(rx_rot, args.array_radius, min_xy, max_xy):
                    continue
                used_angles.append(ang)
                rx_positions.append(rx_rot)

                # 入力順は学習と同じ：[Tx_norm, Rx_norm]
                tx_norm = norm_xy(tx_xy,  min_xy, max_xy)
                rx_norm = norm_xy(rx_rot, min_xy, max_xy)
                total_pos = torch.from_numpy(np.concatenate([tx_norm, rx_norm])[None, None, :]).float().to(device)

                pos_emb   = xyz_emb(total_pos).expand(-1, PIXEL_COUNT_val, -1)
                freq_embd = freq_emb(freqs)
                time_embd = time_emb(times)
                total_in  = torch.cat([pos_emb, freq_embd, time_embd], dim=2)

                # non_norm も [Tx, Rx]
                non_norm_tensor = torch.tensor([tx_xy[0], tx_xy[1], rx_rot[0], rx_rot[1]],
                                               dtype=torch.float32, device=device).view(1, 4)

                outs = []
                P = total_in.shape[1]
                for s in range(0, P, PIXEL_COUNT_val):
                    e = min(P, s + PIXEL_COUNT_val)
                    chunk = total_in[:, s:e, :]
                    if chunk.shape[1] < PIXEL_COUNT_val:
                        pad = torch.zeros(1, PIXEL_COUNT_val - chunk.shape[1], chunk.shape[2], device=device)
                        chunk = torch.cat([chunk, pad], dim=1)
                        out = net(chunk, non_norm_tensor).transpose(1, 2)[:, :, :e - s, :]
                    else:
                        out = net(chunk, non_norm_tensor).transpose(1, 2)
                    outs.append(out)
                out_all = torch.cat(outs, dim=2)  # [1, doa_ch_conf, P, 2]

                arr   = out_all.detach().cpu().numpy()
                mag   = arr[..., 0].reshape(1, doa_ch_conf, dataset.sound_size[1], dataset.sound_size[2])
                phase = arr[..., 1].reshape(1, doa_ch_conf, dataset.sound_size[1], dataset.sound_size[2])

                need_T = mean.shape[-1]
                pad_T  = max(0, need_T - mag.shape[-1])
                if pad_T > 0:
                    mag   = np.pad(mag,   ((0,0),(0,0),(0,0),(0,pad_T)))
                    phase = np.pad(phase, ((0,0),(0,0),(0,0),(0,pad_T)))

                net_mag   = (mag[0] * std + mean)   # (M,F,T)
                net_phase =  phase[0] * phase_std   # (M,F,T)
                net_spec  = get_spectrograms(net_mag, net_phase).astype(np.complex64)  # (M,F,T)
                pred_specs.append(net_spec)

                # DoA（中心[0,0], φ0=π/2）
                mic = pra.beamforming.circular_2D_array(center=[0,0], M=doa_ch_conf, radius=args.array_radius, phi0=math.pi/2)
                doa = pra.doa.algorithms["NormMUSIC"](mic, fs=args.fs, nfft=args.nfft)
                doa.locate_sources(net_spec)
                pred_deg_list.append(int(np.argmax(doa.grid.values)) % 360)

                # 真値角：各 Rx(=rx_rot) から Tx への方向（グローバル座標）
                true_deg = math.degrees(math.atan2(tx_xy[1] - rx_rot[1], tx_xy[0] - rx_rot[0])) % 360.0
                true_deg_list.append(int(true_deg))

            if pred_deg_list:
                pred_arr = np.array(pred_deg_list, dtype=np.int16)
                true_arr = np.array(true_deg_list, dtype=np.int16)
                err = angular_error_deg(pred_arr, true_arr)

                np.savez_compressed(
                    os.path.join(out_dir, f"val_rotate_{val_id:04d}.npz"),
                    pred_sig_spec=np.stack(pred_specs, axis=0),         # (N_rot, M, F, T)
                    position_tx=np.array(tx_xy, dtype=np.float32),      # (2,)
                    position_rx=np.stack(rx_positions, axis=0).astype(np.float32),  # (N_rot, 2)
                    pred_deg=pred_arr, true_deg=true_arr,
                    used_angles=np.array(used_angles, dtype=np.float32),
                    deg_step=float(args.deg_step),
                )
                summary_lines.append(f"{val_id},{len(pred_arr)},{float(err.mean()):.4f}\n")
                all_pred.append(pred_arr); all_true.append(true_arr)

    # ===== dir_ch == 1（8本で1セット剛体回転） =====
    else:
        N = len(dataset.sound_files_val)
        assert N % 8 == 0, f"dir_ch=1 では val 件数({N})が8の倍数である必要があります。"

        group_id = 0
        for g0 in range(0, N, 8):
            tx_list, rx_list, theta0_list, radii = [], [], [], []
            freqs_list, times_list = [], []

            for k in range(8):
                _, _, non_norm_0, freqs_norm, times_norm = dataset.get_item_val(g0 + k)
                nn = non_norm_0.squeeze().cpu().numpy()  # [Tx, Rx]
                tx_xy = nn[:2]
                rx_xy = nn[2:]
                tx_list.append(tx_xy); rx_list.append(rx_xy)

                freqs_list.append(freqs_norm[None].to(device).unsqueeze(2) * (2.0 * math.pi))
                times_list.append(times_norm[None].to(device).unsqueeze(2) * (2.0 * math.pi))
            
            PIXEL_COUNT_val = freqs_list[0].shape[1]

            # 同セットでは Tx は共通と仮定：先頭を代表
            tx_xy = np.array(tx_list[0], dtype=float)
            # 元の8chの幾何中心（回転させるのはこの中心のみ）
            center0 = np.mean(np.stack(rx_list, axis=0), axis=0).astype(float)
            r_center = float(np.linalg.norm(center0 - tx_xy))
            theta_center0 = (math.degrees(math.atan2(center0[1] - tx_xy[1], center0[0] - tx_xy[0])) % 360.0)

            pred_deg_list, true_deg_list, used_angles = [], [], []
            rx_positions_group = []  # (N_rot, 8, 2)
            pred_specs_group   = []  # (N_rot, 8, F, T)

            for d in delta_list:
                # 中心のみを Tx を基準に回転（中心角は θ_center0 + d）
                ang_center = (theta_center0 + d) % 360.0
                center_rot = rotate_about(tx_xy, r_center, ang_center)

                # アレイ配置：ch0 を π/2（上方向）とし、以降は 45° ずつ反時計回りに配置
                base = math.pi / 2.0
                rx_rot_8 = [
                    center_rot + args.array_radius * np.array(
                        [math.cos(base + k * math.pi / 4.0), math.sin(base + k * math.pi / 4.0)],
                        dtype=float
                    )
                    for k in range(8)
                ]

                # 中心が枠内に収まるか（半径ぶんのマージン込み）
                if not array_within_bounds(center_rot, args.array_radius, min_xy, max_xy):
                    continue
                used_angles.append(d)

                specs_8 = []
                for k in range(8):
                    tx_norm = norm_xy(tx_xy,       min_xy, max_xy)
                    rx_norm = norm_xy(rx_rot_8[k], min_xy, max_xy)
                    total_pos = torch.from_numpy(np.concatenate([tx_norm, rx_norm])[None, None, :]).float().to(device)

                    pos_emb   = xyz_emb(total_pos).expand(-1, PIXEL_COUNT_val, -1)
                    freq_embd = freq_emb(freqs_list[k])
                    time_embd = time_emb(times_list[k])
                    total_in  = torch.cat([pos_emb, freq_embd, time_embd], dim=2)

                    non_norm_tensor = torch.tensor([tx_xy[0], tx_xy[1], rx_rot_8[k][0], rx_rot_8[k][1]],
                                                   dtype=torch.float32, device=device).view(1, 4)

                    outs = []
                    P = total_in.shape[1]
                    for s in range(0, P, PIXEL_COUNT_val):
                        e = min(P, s + PIXEL_COUNT_val)
                        chunk = total_in[:, s:e, :]
                        if chunk.shape[1] < PIXEL_COUNT_val:
                            pad = torch.zeros(1, PIXEL_COUNT_val - chunk.shape[1], chunk.shape[2], device=device)
                            chunk = torch.cat([chunk, pad], dim=1)
                            out = net(chunk, non_norm_tensor).transpose(1, 2)[:, :, :e - s, :]
                        else:
                            out = net(chunk, non_norm_tensor).transpose(1, 2)
                        outs.append(out)
                    out_all = torch.cat(outs, dim=2)  # [1,1,P,2]

                    arr   = out_all.detach().cpu().numpy()
                    mag   = arr[..., 0].reshape(1, 1, dataset.sound_size[1], dataset.sound_size[2])
                    phase = arr[..., 1].reshape(1, 1, dataset.sound_size[1], dataset.sound_size[2])

                    need_T = mean.shape[-1]
                    pad_T  = max(0, need_T - mag.shape[-1])
                    if pad_T > 0:
                        mag   = np.pad(mag,   ((0,0),(0,0),(0,0),(0,pad_T)))
                        phase = np.pad(phase, ((0,0),(0,0),(0,0),(0,pad_T)))

                    net_mag   = (mag[0] * std + mean)   # (1,F,T)
                    net_phase =  phase[0] * phase_std   # (1,F,T)
                    spec      = get_spectrograms(net_mag, net_phase)[0].astype(np.complex64)  # (F,T)
                    specs_8.append(spec)

                specs_8 = np.stack(specs_8, axis=0)  # (8,F,T)
                pred_specs_group.append(specs_8)
                rx_positions_group.append(np.stack(rx_rot_8, axis=0))

                mic = pra.beamforming.circular_2D_array(center=[0,0], M=8, radius=args.array_radius, phi0=math.pi/2)
                doa = pra.doa.algorithms["NormMUSIC"](mic, fs=args.fs, nfft=args.nfft)
                doa.locate_sources(specs_8)
                pred_deg_list.append(int(np.argmax(doa.grid.values)) % 360)
                
                rx_center = np.mean(np.stack(rx_rot_8, axis=0), axis=0)
                true_deg  = math.degrees(math.atan2(tx_xy[1] - rx_center[1], tx_xy[0] - rx_center[0])) % 360.0
                true_deg_list.append(int(true_deg))


            if used_angles:
                pred_arr = np.array(pred_deg_list, dtype=np.int16)
                true_arr = np.array(true_deg_list, dtype=np.int16)
                err = angular_error_deg(pred_arr, true_arr)

                np.savez_compressed(
                    os.path.join(out_dir, f"val_group_{group_id:04d}.npz"),
                    pred_sig_spec=np.stack(pred_specs_group, axis=0),     # (N_rot, 8, F, T)
                    position_tx=np.array(tx_xy, dtype=np.float32),        # (2,)
                    position_rx=np.stack(rx_positions_group, axis=0).astype(np.float32),  # (N_rot, 8, 2)
                    pred_deg=pred_arr, true_deg=true_arr,                 # (N_rot,)
                    used_angles=np.array(used_angles, dtype=np.float32),  # Δθ列
                    deg_step=float(args.deg_step),
                )
                summary_lines.append(f"{group_id},{len(used_angles)},{float(err.mean()):.4f}\n")
                all_pred.append(pred_arr); all_true.append(true_arr)
            else:
                summary_lines.append(f"{group_id},0,NaN\n")

            group_id += 1

    # Summary
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.writelines(summary_lines)

    if all_pred:
        pred_all = np.concatenate(all_pred); true_all = np.concatenate(all_true)
        overall = float(angular_error_deg(pred_all, true_all).mean())
        overall_std = float(angular_error_deg(pred_all, true_all).std())
        with open(os.path.join(out_dir, "overall.txt"), "w") as f:
            f.write(f"mean_angular_error_deg={overall:.4f}\n")
            f.write(f"std_angular_error_deg={overall_std:.4f}\n")
        print(f"[DONE] overall mean angular error = {overall:.4f}°")
    else:
        print("[DONE] No usable rotations.")


if __name__ == "__main__":
    main()

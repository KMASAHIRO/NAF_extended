# eval_random_doa.py
# - YAMLのみで設定構成（Options().parse()は使わない）
# - Tx, Rx を [min_xy, max_xy] の範囲で一様ランダムサンプリング（境界からのマージンで内側に制約）
# - dir_ch>1：Rxはアレイ中心扱い（半径 array_radius の円が室内に収まるよう中心をサンプル）
# - dir_ch==1：8本の円形アレイ（半径 array_radius）を、中心をランダムサンプルして配置
# - すべてのサンプルをまとめて 1 ファイル（.npz）に保存
# - predスペクトログラム / Tx座標 / Rx座標 / pred_deg / true_deg / err_deg を格納
# - summary.csv と overall.txt も併せて出力

import os
import math
import yaml
import argparse
import numpy as np
import torch
import pyroomacoustics as pra

from model_pipeline.sound_loader import soundsamples
from model.modules import embedding_module_log
from model.networks import kernel_residual_fc_embeds
from model_pipeline.options import Options


def get_spectrograms(input_stft, input_if):
    """log-mag と IF から複素 STFT を復元する。
    input:
        input_stft: (M,F,T) の log-magnitude
        input_if  : (M,F,T) の instantaneous frequency（ラジアン差分/πの想定）
    output:
        (M,F,T) の complex64
    """
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
    # Options のデフォルトを取得（CLI引数は使わない → 空でparse）
    _op = Options()
    _op.initialize()
    defaults = _op.parser.parse_args([])  # 全デフォルト
    # YAMLで上書き
    for k, v in cfg.items():
        setattr(defaults, k, v)
    return defaults


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


def sample_xy_uniform(min_xy, max_xy, margin=0.0, rng=None):
    """
    矩形[min_xy, max_xy] 内で一様サンプリング。
    margin > 0 のときは四辺から margin だけ内側でサンプル。
    前提: min_xy < max_xy かつ margin が (max_xy - min_xy)/2 より十分小さい。
    """
    if rng is None:
        rng = np.random.default_rng()
    min_xy = np.asarray(min_xy, dtype=float)
    max_xy = np.asarray(max_xy, dtype=float)

    low  = min_xy + margin
    high = max_xy - margin

    if not np.all(high > low):
        raise ValueError(f"margin が大きすぎます: low={low}, high={high}")

    return rng.uniform(low=low, high=high)


@torch.no_grad()
def main():
    # 引数
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       type=str, required=True)
    ap.add_argument("--ckpt",         type=str, required=True)
    ap.add_argument("--out_dir",      type=str, default=None)
    ap.add_argument("--fs",           type=int,   default=16000)
    ap.add_argument("--nfft",         type=int,   default=512)
    ap.add_argument("--array_radius", type=float, default=0.0365)  # 円形アレイ半径
    ap.add_argument("--num_samples",  type=int,   default=500)     # ランダムサンプル数
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    # YAMLだけで opt を作る（Options().parse() は使わない）
    opt = build_opt_from_yaml(args.config)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(opt.save_loc, opt.exp_name)
    out_dir = args.out_dir or os.path.join(exp_dir, "random_eval")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Dataset
    dataset   = soundsamples(opt)
    min_xy    = np.array(dataset.min_pos, dtype=float)
    max_xy    = np.array(dataset.max_pos, dtype=float)
    mean      = dataset.mean.cpu().numpy()   # 形状は (M,F,T) or (1,F,T)
    std       = dataset.std.cpu().numpy()
    phase_std = float(dataset.phase_std)

    # Model
    net, xyz_emb, time_emb, freq_emb = build_model_and_embedders(opt, dataset, device)
    state = torch.load(os.path.expanduser(args.ckpt), map_location=device)
    net.load_state_dict(state["network"])
    net.eval()

    doa_ch_conf = int(opt.dir_ch)

    # freq/time のテンプレート（val の 1 件目から拝借）
    _, _, _, freqs_norm_tmpl, times_norm_tmpl = dataset.get_item_val(0)
    freqs_tmpl = freqs_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
    times_tmpl = times_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
    PIXEL_COUNT_val = freqs_tmpl.shape[1]

    # まとめ保存用のバッファ
    summary_lines = ["sample_id,used,err_deg\n"]

    if doa_ch_conf > 1:
        # (N, M, F, T) を格納
        all_specs = []
        all_tx    = []
        all_rx    = []
        all_pred  = []
        all_true  = []
        all_err   = []

        for sid in range(args.num_samples):
            # Tx は全域から一様
            tx_xy = sample_xy_uniform(min_xy, max_xy, margin=0.0, rng=rng)
            # Rx は「アレイ中心」→ 円が内接するように margin=array_radius
            rx_xy = sample_xy_uniform(min_xy, max_xy, margin=args.array_radius, rng=rng)

            # 入力（埋め込み）
            tx_norm = norm_xy(tx_xy, min_xy, max_xy)
            rx_norm = norm_xy(rx_xy, min_xy, max_xy)
            total_pos = torch.from_numpy(np.concatenate([tx_norm, rx_norm])[None, None, :]).float().to(device)

            pos_emb   = xyz_emb(total_pos).expand(-1, PIXEL_COUNT_val, -1)
            freq_embd = freq_emb(freqs_tmpl)
            time_embd = time_emb(times_tmpl)
            total_in  = torch.cat([pos_emb, freq_embd, time_embd], dim=2)

            non_norm_tensor = torch.tensor([tx_xy[0], tx_xy[1], rx_xy[0], rx_xy[1]],
                                           dtype=torch.float32, device=device).view(1, 4)

            # ネット推論（P を分割する一般形のまま）
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
            out_all = torch.cat(outs, dim=2)  # [1, M, P, 2]

            arr   = out_all.detach().cpu().numpy()
            mag   = arr[..., 0].reshape(1, doa_ch_conf, dataset.sound_size[1], dataset.sound_size[2])
            phase = arr[..., 1].reshape(1, doa_ch_conf, dataset.sound_size[1], dataset.sound_size[2])

            # T 次元を学習時統計に合わせる（padding）
            need_T = mean.shape[-1]
            pad_T  = max(0, need_T - mag.shape[-1])
            if pad_T > 0:
                mag   = np.pad(mag,   ((0,0),(0,0),(0,0),(0,pad_T)))
                phase = np.pad(phase, ((0,0),(0,0),(0,0),(0,pad_T)))

            net_mag   = (mag[0] * std + mean)   # (M,F,T)
            net_phase =  phase[0] * phase_std   # (M,F,T)
            net_spec  = get_spectrograms(net_mag, net_phase).astype(np.complex64)  # (M,F,T)

            # DoA 推定（M=dir_ch_conf の円形アレイ）
            mic = pra.beamforming.circular_2D_array(center=[0,0], M=doa_ch_conf, radius=args.array_radius, phi0=math.pi/2)
            doa = pra.doa.algorithms["NormMUSIC"](mic, fs=args.fs, nfft=args.nfft)
            doa.locate_sources(net_spec)
            pred_deg = int(np.argmax(doa.grid.values)) % 360

            # 真値角：Rx(中心) → Tx
            true_deg = int(math.degrees(math.atan2(tx_xy[1] - rx_xy[1], tx_xy[0] - rx_xy[0])) % 360.0)
            err = float(angular_error_deg(pred_deg, true_deg))

            # 蓄積
            all_specs.append(net_spec)            # (M,F,T)
            all_tx.append(tx_xy)                  # (2,)
            all_rx.append(rx_xy)                  # (2,)
            all_pred.append(pred_deg)
            all_true.append(true_deg)
            all_err.append(err)
            summary_lines.append(f"{sid},1,{err:.4f}\n")

        # まとめて保存
        np.savez_compressed(
            os.path.join(out_dir, "random_eval_all.npz"),
            pred_sig_spec=np.stack(all_specs, axis=0),      # (N, M, F, T)
            position_tx=np.stack(all_tx, axis=0).astype(np.float32),  # (N, 2)
            position_rx=np.stack(all_rx, axis=0).astype(np.float32),  # (N, 2)
            pred_deg=np.array(all_pred, dtype=np.int16),    # (N,)
            true_deg=np.array(all_true, dtype=np.int16),    # (N,)
            err_deg=np.array(all_err, dtype=np.float32),    # (N,)
        )

    else:
        # dir_ch==1 の場合（8本の仮想円形アレイで DoA）
        all_specs = []
        all_tx    = []
        all_rx    = []  # (8,2)
        all_pred  = []
        all_true  = []
        all_err   = []

        for sid in range(args.num_samples):
            # Tx は全域から一様
            tx_xy = sample_xy_uniform(min_xy, max_xy, margin=0.0, rng=rng)
            # アレイ中心は margin=array_radius で一様
            center_xy = sample_xy_uniform(min_xy, max_xy, margin=args.array_radius, rng=rng)

            # 8ch 円形配置（ch0: π/2 上向き、以降 45° 刻み）
            base = math.pi / 2.0
            rx_list = [
                center_xy + args.array_radius * np.array(
                    [math.cos(base + k * math.pi / 4.0), math.sin(base + k * math.pi / 4.0)],
                    dtype=float
                )
                for k in range(8)
            ]

            # スペクトログラム推定（各ch）
            _, _, _, freqs_norm_tmpl, times_norm_tmpl = dataset.get_item_val(0)
            freqs_tmpl = freqs_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
            times_tmpl = times_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
            PIXEL_COUNT_val = freqs_tmpl.shape[1]

            specs_8 = []
            for k in range(8):
                tx_norm = norm_xy(tx_xy,      min_xy, max_xy)
                rx_norm = norm_xy(rx_list[k], min_xy, max_xy)
                total_pos = torch.from_numpy(np.concatenate([tx_norm, rx_norm])[None, None, :]).float().to(device)

                pos_emb   = xyz_emb(total_pos).expand(-1, PIXEL_COUNT_val, -1)
                freq_embd = freq_emb(freqs_tmpl)
                time_embd = time_emb(times_tmpl)
                total_in  = torch.cat([pos_emb, freq_embd, time_embd], dim=2)

                non_norm_tensor = torch.tensor([tx_xy[0], tx_xy[1], rx_list[k][0], rx_list[k][1]],
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

            specs_8 = np.stack(specs_8, axis=0)  # (8, F, T)

            # DoA（8ch 円形アレイ）
            mic = pra.beamforming.circular_2D_array(center=[0,0], M=8, radius=args.array_radius, phi0=math.pi/2)
            doa = pra.doa.algorithms["NormMUSIC"](mic, fs=args.fs, nfft=args.nfft)
            doa.locate_sources(specs_8)
            pred_deg = int(np.argmax(doa.grid.values)) % 360

            # 真値角：アレイ中心 → Tx
            true_deg = int(math.degrees(math.atan2(tx_xy[1] - center_xy[1], tx_xy[0] - center_xy[0])) % 360.0)
            err = float(angular_error_deg(pred_deg, true_deg))

            # 蓄積
            all_specs.append(specs_8)                    # (8,F,T)
            all_tx.append(tx_xy)                         # (2,)
            all_rx.append(np.stack(rx_list, axis=0))     # (8,2)
            all_pred.append(pred_deg)
            all_true.append(true_deg)
            all_err.append(err)
            summary_lines.append(f"{sid},1,{err:.4f}\n")

        # まとめて保存
        np.savez_compressed(
            os.path.join(out_dir, "random_eval_all.npz"),
            pred_sig_spec=np.stack(all_specs, axis=0),                # (N, 8, F, T)
            position_tx=np.stack(all_tx, axis=0).astype(np.float32),  # (N, 2)
            position_rx=np.stack(all_rx, axis=0).astype(np.float32),  # (N, 8, 2)
            pred_deg=np.array(all_pred, dtype=np.int16),              # (N,)
            true_deg=np.array(all_true, dtype=np.int16),              # (N,)
            err_deg=np.array(all_err, dtype=np.float32),              # (N,)
        )

    # Summary ファイル
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.writelines(summary_lines)

    # overall
    if len(summary_lines) > 1:
        overall = float(np.mean([float(line.strip().split(",")[-1]) for line in summary_lines[1:]]))
        with open(os.path.join(out_dir, "overall.txt"), "w") as f:
            f.write(f"mean_angular_error_deg={overall:.4f}\n")
        print(f"[DONE] overall mean angular error = {overall:.4f}°")
    else:
        print("[DONE] No usable samples.")


if __name__ == "__main__":
    main()

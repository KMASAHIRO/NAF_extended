#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_circular_sweep_doa.py

- YAMLのみで設定（Options().parse()は使わない）
- Txは固定8点:
    (1, 2.5),
    (2, 2.5),
    (2, 5.5),
    (3, 6.5),
    (3, 2.5),
    (3, 5.5),
    (2, 6.5),
    (4, 2.5)
- Rxは x=1..4, y=1.5..6.5 のgrid上（Txと重なる位置を除く）
- dir_ch>1: Rxはアレイ中心。円形アレイ(半径 array_radius)が室内に収まる点のみ採用
- dir_ch==1: 各Rx中心に8本の仮想円形アレイを配置してDoA
- すべて1ファイル circular_eval_all.npz にまとめて保存
"""

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
    """log-mag と IF から複素 STFT を復元する。(M,F,T)->complex64"""
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
    _op = Options()
    _op.initialize()
    defaults = _op.parser.parse_args([])  # 全デフォルト
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


def array_within_bounds(center_xy, radius, min_xy, max_xy):
    """円（center, radius）が矩形[min_xy,max_xy]に完全内包されるか"""
    x, y = center_xy
    return (
        (x - radius) >= min_xy[0]
        and (y - radius) >= min_xy[1]
        and (x + radius) <= max_xy[0]
        and (y + radius) <= max_xy[1]
    )


def write_overall_from_errors(err_array, out_dir, tag=""):
    """誤差配列から mean / std を計算して overall.txt に書き出すユーティリティ"""
    err_array = np.asarray(err_array, dtype=np.float32)
    mean_err = float(np.mean(err_array))
    std_err  = float(np.std(err_array))
    overall_path = os.path.join(out_dir, "overall.txt")
    with open(overall_path, "w") as f:
        f.write(f"mean_angular_error_deg={mean_err:.4f}\n")
        f.write(f"std_angular_error_deg={std_err:.4f}\n")
    if tag:
        print(f"[{tag}] overall mean angular error = {mean_err:.4f}°, std = {std_err:.4f}°")
    else:
        print(f"[DONE] overall mean angular error = {mean_err:.4f}°, std = {std_err:.4f}°")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       type=str, required=False)
    ap.add_argument("--ckpt",         type=str, required=False)
    ap.add_argument("--out_dir",      type=str, default=None)
    ap.add_argument("--fs",           type=int,   default=16000)
    ap.add_argument("--nfft",         type=int,   default=512)
    ap.add_argument("--array_radius", type=float, default=0.0365)  # 円形アレイ半径
    # ★追加: 既存結果をロードして overall.txt だけ出力するモード
    ap.add_argument("--load_only",    action="store_true",
                    help="既存の circular_eval_all.npz をロードして統計だけ出力する")
    args = ap.parse_args()

    # === load_only モード: 推論は一切せず、npz から err_deg を読んで mean/std を出すだけ ===
    if args.load_only:
        if args.out_dir is None:
            raise ValueError("--load_only を使う場合は --out_dir を指定してください。")
        out_dir = args.out_dir
        npz_path = os.path.join(out_dir, "circular_eval_all.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"npz が見つかりません: {npz_path}")
        data = np.load(npz_path)
        if "err_deg" not in data:
            raise KeyError("circular_eval_all.npz に 'err_deg' が含まれていません。")
        err_array = data["err_deg"]
        write_overall_from_errors(err_array, out_dir, tag="LOAD_ONLY")
        return

    # ここから先は通常モード（推論を走らせる）

    if args.config is None or args.ckpt is None:
        raise ValueError("--config と --ckpt は通常モードでは必須です。")

    # 固定Tx座標（8点）
    tx_list = np.array([
        [1.0, 2.5],
        [2.0, 2.5],
        [2.0, 5.5],
        [3.0, 6.5],
        [3.0, 2.5],
        [3.0, 5.5],
        [2.0, 6.5],
        [4.0, 2.5],
    ], dtype=float)

    # Rx grid座標: x=1..4, y=1.5..6.5 (ステップ1.0), Txと重なる点を除外
    rx_x_vals = np.arange(1.0, 4.0 + 1e-6, 1.0)
    rx_y_vals = np.arange(1.5, 6.5 + 1e-6, 1.0)

    tx_set = {(float(x), float(y)) for x, y in tx_list}
    rx_grid = []
    for x in rx_x_vals:
        for y in rx_y_vals:
            rx_grid.append(np.array([x, y], dtype=float))
    rx_grid = np.array(rx_grid, dtype=float)  # (N_rx, 2)

    # YAMLで opt 構築
    opt = build_opt_from_yaml(args.config)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(opt.save_loc, opt.exp_name)
    out_dir = args.out_dir or os.path.join(exp_dir, "circular_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Dataset & 正規化定数
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

    # freq/time のテンプレート（val から拝借）
    _, _, _, freqs_norm_tmpl, times_norm_tmpl = dataset.get_item_val(0)
    freqs_tmpl = freqs_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
    times_tmpl = times_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
    PIXEL_COUNT_val = freqs_tmpl.shape[1]

    # 集約バッファ
    summary_lines = ["idx,tx_x,tx_y,rx_x,rx_y,used,err_deg\n"]

    if doa_ch_conf > 1:
        # dir_ch > 1: (N, M, F, T)
        all_specs, all_tx, all_rx, all_pred, all_true, all_err = [], [], [], [], [], []
        sid = 0

        for tx_xy in tx_list:
            for rx_center in rx_grid:
                # ★同じ検証内で tx と rx が同一座標ならスキップ
                if np.allclose(rx_center, tx_xy, atol=1e-6):
                    continue

                # アレイ中心が室内に収まるか（円形アレイが内包されるか）
                if not array_within_bounds(rx_center, args.array_radius, min_xy, max_xy):
                    summary_lines.append(
                        f"{sid},{tx_xy[0]},{tx_xy[1]},{rx_center[0]},{rx_center[1]},0,NaN\n"
                    )
                    sid += 1
                    continue

                # 入力（[Tx_norm, Rx_norm]）
                tx_norm = norm_xy(tx_xy,    min_xy, max_xy)
                rx_norm = norm_xy(rx_center, min_xy, max_xy)
                total_pos = torch.from_numpy(
                    np.concatenate([tx_norm, rx_norm])[None, None, :]
                ).float().to(device)

                pos_emb   = xyz_emb(total_pos).expand(-1, PIXEL_COUNT_val, -1)
                freq_embd = freq_emb(freqs_tmpl)
                time_embd = time_emb(times_tmpl)
                total_in  = torch.cat([pos_emb, freq_embd, time_embd], dim=2)

                non_norm_tensor = torch.tensor(
                    [tx_xy[0], tx_xy[1], rx_center[0], rx_center[1]],
                    dtype=torch.float32, device=device
                ).view(1, 4)

                # 推論（分割一般形）
                outs = []
                P = total_in.shape[1]
                for s in range(0, P, PIXEL_COUNT_val):
                    e = min(P, s + PIXEL_COUNT_val)
                    chunk = total_in[:, s:e, :]
                    if chunk.shape[1] < PIXEL_COUNT_val:
                        pad = torch.zeros(
                            1, PIXEL_COUNT_val - chunk.shape[1], chunk.shape[2],
                            device=device
                        )
                        chunk = torch.cat([chunk, pad], dim=1)
                        out = net(chunk, non_norm_tensor).transpose(1, 2)[:, :, :e - s, :]
                    else:
                        out = net(chunk, non_norm_tensor).transpose(1, 2)
                    outs.append(out)
                out_all = torch.cat(outs, dim=2)  # [1, M, P, 2]

                arr   = out_all.detach().cpu().numpy()
                mag   = arr[..., 0].reshape(1, doa_ch_conf, dataset.sound_size[1], dataset.sound_size[2])
                phase = arr[..., 1].reshape(1, doa_ch_conf, dataset.sound_size[1], dataset.sound_size[2])

                # 学習統計にT次元合わせ
                need_T = mean.shape[-1]
                pad_T  = max(0, need_T - mag.shape[-1])
                if pad_T > 0:
                    mag   = np.pad(mag,   ((0,0),(0,0),(0,0),(0,pad_T)))
                    phase = np.pad(phase, ((0,0),(0,0),(0,0),(0,pad_T)))

                net_mag   = (mag[0] * std + mean)   # (M,F,T)
                net_phase =  phase[0] * phase_std   # (M,F,T)
                net_spec  = get_spectrograms(net_mag, net_phase).astype(np.complex64)  # (M,F,T)

                # DoA (実マイク本数 = doa_ch_conf)
                mic = pra.beamforming.circular_2D_array(
                    center=[0,0], M=doa_ch_conf, radius=args.array_radius, phi0=math.pi/2
                )
                doa = pra.doa.algorithms["NormMUSIC"](mic, fs=args.fs, nfft=args.nfft)
                doa.locate_sources(net_spec)
                pred_deg = int(np.argmax(doa.grid.values)) % 360

                # 真値角：Rx中心 -> Tx
                true_deg = int(
                    math.degrees(
                        math.atan2(tx_xy[1] - rx_center[1], tx_xy[0] - rx_center[0])
                    ) % 360.0
                )
                err = float(angular_error_deg(pred_deg, true_deg))

                # 蓄積
                all_specs.append(net_spec)
                all_tx.append(tx_xy)
                all_rx.append(rx_center)
                all_pred.append(pred_deg)
                all_true.append(true_deg)
                all_err.append(err)
                summary_lines.append(
                    f"{sid},{tx_xy[0]},{tx_xy[1]},{rx_center[0]},{rx_center[1]},1,{err:.4f}\n"
                )
                sid += 1

        # 保存
        np.savez_compressed(
            os.path.join(out_dir, "circular_eval_all.npz"),
            pred_sig_spec=np.stack(all_specs, axis=0),                 # (N, M, F, T)
            position_tx=np.stack(all_tx, axis=0).astype(np.float32),   # (N, 2)
            position_rx=np.stack(all_rx, axis=0).astype(np.float32),   # (N, 2) アレイ中心
            pred_deg=np.array(all_pred, dtype=np.int16),               # (N,)
            true_deg=np.array(all_true, dtype=np.int16),               # (N,)
            err_deg=np.array(all_err, dtype=np.float32),               # (N,)
        )

    else:
        # dir_ch == 1（8本の仮想円形アレイ）
        all_specs, all_tx, all_rx, all_pred, all_true, all_err = [], [], [], [], [], []
        sid = 0

        # テンプレ再取得（念のため）
        _, _, _, freqs_norm_tmpl, times_norm_tmpl = dataset.get_item_val(0)
        freqs_tmpl = freqs_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
        times_tmpl = times_norm_tmpl[None].to(device).unsqueeze(2) * (2.0 * math.pi)
        PIXEL_COUNT_val = freqs_tmpl.shape[1]

        for tx_xy in tx_list:
            for center_xy in rx_grid:
                # ★同じ検証内で tx と rx(center) が同一座標ならスキップ
                if np.allclose(center_xy, tx_xy, atol=1e-6):
                    continue

                # アレイが室内に収まるか
                if not array_within_bounds(center_xy, args.array_radius, min_xy, max_xy):
                    summary_lines.append(
                        f"{sid},{tx_xy[0]},{tx_xy[1]},{center_xy[0]},{center_xy[1]},0,NaN\n"
                    )
                    sid += 1
                    continue

                # 8ch配置（ch0=π/2上向き、45°刻み）
                base = math.pi / 2.0
                rx_list = [
                    center_xy + args.array_radius * np.array(
                        [math.cos(base + k * math.pi / 4.0),
                         math.sin(base + k * math.pi / 4.0)],
                        dtype=float
                    )
                    for k in range(8)
                ]

                specs_8 = []
                for k in range(8):
                    tx_norm = norm_xy(tx_xy,      min_xy, max_xy)
                    rx_norm = norm_xy(rx_list[k], min_xy, max_xy)
                    total_pos = torch.from_numpy(
                        np.concatenate([tx_norm, rx_norm])[None, None, :]
                    ).float().to(device)

                    pos_emb   = xyz_emb(total_pos).expand(-1, PIXEL_COUNT_val, -1)
                    freq_embd = freq_emb(freqs_tmpl)
                    time_embd = time_emb(times_tmpl)
                    total_in  = torch.cat([pos_emb, freq_embd, time_embd], dim=2)

                    non_norm_tensor = torch.tensor(
                        [tx_xy[0], tx_xy[1], rx_list[k][0], rx_list[k][1]],
                        dtype=torch.float32, device=device
                    ).view(1, 4)

                    outs = []
                    P = total_in.shape[1]
                    for s in range(0, P, PIXEL_COUNT_val):
                        e = min(P, s + PIXEL_COUNT_val)
                        chunk = total_in[:, s:e, :]
                        if chunk.shape[1] < PIXEL_COUNT_val:
                            pad = torch.zeros(
                                1, PIXEL_COUNT_val - chunk.shape[1], chunk.shape[2],
                                device=device
                            )
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

                # DoA
                mic = pra.beamforming.circular_2D_array(
                    center=[0,0], M=8, radius=args.array_radius, phi0=math.pi/2
                )
                doa = pra.doa.algorithms["NormMUSIC"](mic, fs=args.fs, nfft=args.nfft)
                doa.locate_sources(specs_8)
                pred_deg = int(np.argmax(doa.grid.values)) % 360

                # 真値角：アレイ中心 -> Tx
                true_deg = int(
                    math.degrees(
                        math.atan2(tx_xy[1] - center_xy[1], tx_xy[0] - center_xy[0])
                    ) % 360.0
                )
                err = float(angular_error_deg(pred_deg, true_deg))

                # 蓄積
                all_specs.append(specs_8)
                all_tx.append(tx_xy)
                all_rx.append(np.stack(rx_list, axis=0))
                all_pred.append(pred_deg)
                all_true.append(true_deg)
                all_err.append(err)
                summary_lines.append(
                    f"{sid},{tx_xy[0]},{tx_xy[1]},{center_xy[0]},{center_xy[1]},1,{err:.4f}\n"
                )
                sid += 1

        np.savez_compressed(
            os.path.join(out_dir, "circular_eval_all.npz"),
            pred_sig_spec=np.stack(all_specs, axis=0),                 # (N, 8, F, T)
            position_tx=np.stack(all_tx, axis=0).astype(np.float32),   # (N, 2)
            position_rx=np.stack(all_rx, axis=0).astype(np.float32),   # (N, 8, 2)
            pred_deg=np.array(all_pred, dtype=np.int16),               # (N,)
            true_deg=np.array(all_true, dtype=np.int16),               # (N,)
            err_deg=np.array(all_err, dtype=np.float32),               # (N,)
        )

    # Summary と overall
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.writelines(summary_lines)

    used_rows = [line for line in summary_lines[1:] if ",1," in line]
    if used_rows:
        # used==1 の行の err_deg を取り出して mean/std を計算
        err_vals = [float(line.strip().split(",")[-1]) for line in used_rows]
        write_overall_from_errors(err_vals, out_dir)
    else:
        print("[DONE] No usable samples (all out-of-bounds).")


if __name__ == "__main__":
    main()

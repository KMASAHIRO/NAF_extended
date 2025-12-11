#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NAF IR metrics evaluation

- NAFのnpz (pred_sig_spec / ori_sig_spec) からIR波形を復元
- 各チャンネル(全S×M本)に対して以下の誤差を計算:
    angle, amp, env(%), T60(%), C50(dB), EDT(ms)
- それぞれの mean / std をまとめて txt / pkl に出力
"""

import os
import argparse
import pickle

import numpy as np
import torch
import librosa

from scipy import stats
from scipy.signal import hilbert
import scipy
import auraloss


# ==========================================================
# T60 / EDT 計算（元実装の式そのまま）
# ==========================================================

def t60_EDT_cal(energys, init_db=-5, end_db=-25, factor=3.0, fs=16000):
    """
    energys : (N, T) の dB エネルギー
    戻り値 : t60_all, edt_all （どちらも [秒]）
    """
    t60_all = []
    edt_all = []

    for energy in energys:
        # === EDT: 0 dB → -10 dB ===
        edt_factor = 6.0
        energy_n10db = energy[np.abs(energy - (-10)).argmin()]
        n10db_sample = np.where(energy == energy_n10db)[0][0]
        edt = n10db_sample / fs * edt_factor  # [秒]

        # === T60: init_db ～ end_db を直線近似 ===
        energy_init = energy[np.abs(energy - init_db).argmin()]
        energy_end  = energy[np.abs(energy - end_db).argmin()]
        init_sample = np.where(energy == energy_init)[0][0]
        end_sample = np.where(energy == energy_end)[0][0]

        x = np.arange(init_sample, end_sample + 1) / fs
        y = energy[init_sample:end_sample + 1]

        slope, intercept = stats.linregress(x, y)[0:2]

        db_regress_init = (init_db - intercept) / slope
        db_regress_end  = (end_db  - intercept) / slope

        t60 = factor * (db_regress_end - db_regress_init)  # [秒]

        t60_all.append(t60)
        edt_all.append(edt)

    return np.array(t60_all), np.array(edt_all)


# ==========================================================
# メトリクス計算本体（元式＋単位変更のみ）
# ==========================================================

def metric_cal(ori_ir, pred_ir, fs=16000, window=32):
    """
    ori_ir, pred_ir : (N, T) または (T,)

    返す値と単位:
      angle_mean / std : 無次元
      amp_mean   / std : 無次元
      env_mean   / std : [%]
      t60_mean   / std : [%]
      C50_mean   / std : [dB]
      EDT_mean   / std : [ms]
    """

    if ori_ir.ndim == 1:
        ori_ir = ori_ir[np.newaxis, :]
    if pred_ir.ndim == 1:
        pred_ir = pred_ir[np.newaxis, :]

    # === STFT loss（元実装通り計算するが返さない） ===
    multi_stft = auraloss.freq.MultiResolutionSTFTLoss(
        w_lin_mag=1,
        fft_sizes=[512, 256, 128],
        win_lengths=[300, 150, 75],
        hop_sizes=[60, 30, 8],
    )
    _ = multi_stft(torch.tensor(ori_ir).unsqueeze(1),
                   torch.tensor(pred_ir).unsqueeze(1))

    # ========= 1) angle error =========
    fft_ori     = np.fft.fft(ori_ir, axis=-1)
    fft_predict = np.fft.fft(pred_ir, axis=-1)

    cos_ori  = np.cos(np.angle(fft_ori))
    cos_pred = np.cos(np.angle(fft_predict))
    sin_ori  = np.sin(np.angle(fft_ori))
    sin_pred = np.sin(np.angle(fft_predict))

    angle_diff = np.abs(cos_ori - cos_pred) + np.abs(sin_ori - sin_pred)
    angle_error_mean = angle_diff.mean()
    angle_error_std  = angle_diff.std()

    # ========= 2) amplitude error（元式そのまま） =========
    amp_ori = scipy.ndimage.convolve1d(
        np.abs(fft_ori), np.ones(window), axis=-1
    )
    amp_predict = scipy.ndimage.convolve1d(
        np.abs(fft_predict), np.ones(window), axis=-1
    )

    amp_rel_diff = np.abs(amp_ori - amp_predict) / amp_ori
    amp_error_mean = amp_rel_diff.mean()
    amp_error_std  = amp_rel_diff.std()

    # ========= 3) envelope error（%） =========
    ori_env  = np.abs(hilbert(ori_ir))
    pred_env = np.abs(hilbert(pred_ir))

    env_rel_diff = np.abs(ori_env - pred_env) / np.max(ori_env, axis=1, keepdims=True)
    env_error_vals = env_rel_diff * 100.0  # [%]
    env_error_mean = env_error_vals.mean()
    env_error_std  = env_error_vals.std()

    # ========= 4) energy, T60(%), EDT(ms) =========
    ori_energy = 10.0 * np.log10(
        np.cumsum(ori_ir[:, ::-1]**2 + 1e-9, axis=-1)[:, ::-1]
    )
    pred_energy = 10.0 * np.log10(
        np.cumsum(pred_ir[:, ::-1]**2 + 1e-9, axis=-1)[:, ::-1]
    )

    ori_energy  -= ori_energy[:, 0].reshape(-1, 1)
    pred_energy -= pred_energy[:, 0].reshape(-1, 1)

    ori_t60, ori_edt   = t60_EDT_cal(ori_energy, fs=fs)
    pred_t60, pred_edt = t60_EDT_cal(pred_energy, fs=fs)

    # T60 相対誤差 [%]
    t60_rel_err    = np.abs(ori_t60 - pred_t60) / ori_t60
    t60_error_vals = t60_rel_err * 100.0
    t60_error_mean = t60_error_vals.mean()
    t60_error_std  = t60_error_vals.std()

    # EDT 絶対誤差 [ms]
    edt_abs_err    = np.abs(ori_edt - pred_edt)
    edt_error_vals = edt_abs_err * 1000.0
    edt_error_mean = edt_error_vals.mean()
    edt_error_std  = edt_error_vals.std()

    # ========= 5) C50 (dB) =========
    base_sample  = 0
    samples_50ms = int(0.05 * fs)

    energy_ori_early = np.sum(ori_ir[:, base_sample:samples_50ms]**2, axis=-1)
    energy_ori_late  = np.sum(ori_ir[:, samples_50ms:]**2, axis=-1)
    energy_pred_early = np.sum(pred_ir[:, base_sample:samples_50ms]**2, axis=-1)
    energy_pred_late  = np.sum(pred_ir[:, samples_50ms:]**2, axis=-1)

    C50_ori  = 10.0 * np.log10(energy_ori_early / energy_ori_late)
    C50_pred = 10.0 * np.log10(energy_pred_early / energy_pred_late)

    C50_abs_err = np.abs(C50_ori - C50_pred)
    C50_error_mean = C50_abs_err.mean()
    C50_error_std  = C50_abs_err.std()

    return (
        angle_error_mean,
        amp_error_mean,
        env_error_mean,   # [%]
        t60_error_mean,   # [%]
        C50_error_mean,   # [dB]
        edt_error_mean,   # [ms]
        angle_error_std,
        amp_error_std,
        env_error_std,    # [%]
        t60_error_std,    # [%]
        C50_error_std,
        edt_error_std,    # [ms]
    )


# ==========================================================
# NAFの npz から IR を生成
# ==========================================================

def istft_ir(spec: np.ndarray, nfft=512, hop=128, win="hann") -> np.ndarray:
    """
    spec: (G, F, T) complex
    return: (G, Nt)
    （あなたのホワイトノイズコードの istft_ir と同じ仕様）
    """
    y = librosa.istft(
        spec,
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=win,
        center=True,
    )
    return y.astype(np.float32)


def load_ir_from_naf_npz(npz_path: str,
                         nfft_ir: int = 512,
                         hop_ir: int = 128,
                         win_ir: str = "hann"):
    """
    NAF用 npz から IR 波形を取り出す。

    期待するキー:
      pred_sig_spec : (S, M, F, T) or (M, F, T)
      ori_sig_spec  : 同形状

    戻り値:
      ori_ir  : (N, Nt)
      pred_ir : (N, Nt)
      （N = S×M）
    """
    data = np.load(npz_path)

    if "pred_sig_spec" not in data or "ori_sig_spec" not in data:
        raise KeyError(f"{npz_path} に pred_sig_spec / ori_sig_spec がありません。")

    pred_spec = np.asarray(data["pred_sig_spec"])
    ori_spec  = np.asarray(data["ori_sig_spec"])

    # (M,F,T) → (1,M,F,T) に揃える
    if pred_spec.ndim == 3:
        pred_spec = pred_spec[None, ...]
        ori_spec  = ori_spec[None, ...]

    if pred_spec.ndim != 4:
        raise ValueError(f"Unexpected pred_sig_spec ndim={pred_spec.ndim}, "
                         f"expected 3 or 4.")

    S, M, F, T = pred_spec.shape
    if ori_spec.shape != (S, M, F, T):
        raise ValueError(f"pred_sig_spec shape {pred_spec.shape} と "
                         f"ori_sig_spec shape {ori_spec.shape} が一致しません。")

    ori_list  = []
    pred_list = []

    for s in range(S):
        # spec_s: (M,F,T)
        spec_pred_s = pred_spec[s]
        spec_ori_s  = ori_spec[s]

        ir_pred_s = istft_ir(spec_pred_s, nfft=nfft_ir, hop=hop_ir, win=win_ir)  # (M,Nt)
        ir_ori_s  = istft_ir(spec_ori_s,  nfft=nfft_ir, hop=hop_ir, win=win_ir)  # (M,Nt)

        if ir_pred_s.shape != ir_ori_s.shape:
            raise ValueError(f"ISTFT後の shape が一致しません: pred {ir_pred_s.shape}, ori {ir_ori_s.shape}")

        pred_list.append(ir_pred_s)
        ori_list.append(ir_ori_s)

    # (S,M,Nt) → (S*M,Nt)
    pred_ir = np.concatenate(pred_list, axis=0)
    ori_ir  = np.concatenate(ori_list,  axis=0)

    return ori_ir, pred_ir


# ==========================================================
# メイン処理
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute IR metrics (NAF, pred_sig_spec / ori_sig_spec) from a npz file."
    )
    parser.add_argument("--npz_path", required=True,
                        help="NAF npz file (must contain pred_sig_spec and ori_sig_spec).")
    parser.add_argument("--outdir", required=True,
                        help="Directory to save metrics txt and pkl.")
    parser.add_argument("--fs", type=int, default=16000,
                        help="Sampling rate used in metric_cal (default: 16000).")
    parser.add_argument("--window", type=int, default=32,
                        help="Amplitude smoothing window size (default: 32).")
    # IR用ISTFTパラメータ（必要なら変更可能だが、既定はホワイトノイズコードに合わせる）
    parser.add_argument("--ir_nfft", type=int, default=512,
                        help="nfft used for ISTFT of IR (default: 512).")
    parser.add_argument("--ir_hop", type=int, default=128,
                        help="hop length used for ISTFT of IR (default: 128).")
    parser.add_argument("--ir_win", type=str, default="hann",
                        help="window used for ISTFT of IR (default: 'hann').")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- NAF npz から IR 波形を取り出し ---
    ori_ir, pred_ir = load_ir_from_naf_npz(
        args.npz_path,
        nfft_ir=args.ir_nfft,
        hop_ir=args.ir_hop,
        win_ir=args.ir_win,
    )

    # --- メトリクス計算 ---
    (
        angle_mean,
        amp_mean,
        env_mean,
        t60_mean,
        c50_mean,
        edt_mean,
        angle_std,
        amp_std,
        env_std,
        t60_std,
        c50_std,
        edt_std,
    ) = metric_cal(ori_ir, pred_ir, fs=args.fs, window=args.window)

    metrics = {
        "npz_path": args.npz_path,
        "fs": args.fs,
        "window": args.window,
        "num_signals": int(ori_ir.shape[0]),
        "angle_mean": float(angle_mean),
        "amp_mean": float(amp_mean),
        "env_mean_percent": float(env_mean),      # [%]
        "t60_mean_percent": float(t60_mean),      # [%]
        "C50_mean_db": float(c50_mean),           # [dB]
        "EDT_mean_ms": float(edt_mean),           # [ms]
        "angle_std": float(angle_std),
        "amp_std": float(amp_std),
        "env_std_percent": float(env_std),        # [%]
        "t60_std_percent": float(t60_std),        # [%]
        "C50_std_db": float(c50_std),             # [dB]
        "EDT_std_ms": float(edt_std),             # [ms]
        "ir_nfft": args.ir_nfft,
        "ir_hop": args.ir_hop,
        "ir_win": args.ir_win,
    }

    base = os.path.splitext(os.path.basename(args.npz_path))[0]
    txt_path = os.path.join(args.outdir, f"{base}_naf_metrics.txt")
    pkl_path = os.path.join(args.outdir, f"{base}_naf_metrics.pkl")

    # --- txt 出力 ---
    lines = []
    lines.append("# NAF IR evaluation metrics\n")
    lines.append(f"npz_path   : {args.npz_path}\n")
    lines.append(f"fs         : {args.fs}\n")
    lines.append(f"window     : {args.window}\n")
    lines.append(f"ir_nfft    : {args.ir_nfft}\n")
    lines.append(f"ir_hop     : {args.ir_hop}\n")
    lines.append(f"ir_win     : {args.ir_win}\n")
    lines.append(f"num_signals: {metrics['num_signals']}\n\n")
    lines.append("Metric order: angle, amp, env(%), t60(%), C50(dB), EDT(ms)\n\n")

    lines.append("=== Mean errors ===\n")
    lines.append(f"angle_mean         : {angle_mean:.6f}\n")
    lines.append(f"amp_mean           : {amp_mean:.6f}\n")
    lines.append(f"env_mean_percent   : {env_mean:.6f}\n")
    lines.append(f"t60_mean_percent   : {t60_mean:.6f}\n")
    lines.append(f"C50_mean_db        : {c50_mean:.6f}\n")
    lines.append(f"EDT_mean_ms        : {edt_mean:.6f}\n\n")

    lines.append("=== Std of errors ===\n")
    lines.append(f"angle_std          : {angle_std:.6f}\n")
    lines.append(f"amp_std            : {amp_std:.6f}\n")
    lines.append(f"env_std_percent    : {env_std:.6f}\n")
    lines.append(f"t60_std_percent    : {t60_std:.6f}\n")
    lines.append(f"C50_std_db         : {c50_std:.6f}\n")
    lines.append(f"EDT_std_ms         : {edt_std:.6f}\n")

    with open(txt_path, "w") as f:
        f.writelines(lines)

    # --- pkl 出力 ---
    with open(pkl_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"[INFO] Saved txt  -> {txt_path}")
    print(f"[INFO] Saved pkl  -> {pkl_path}")


if __name__ == "__main__":
    main()

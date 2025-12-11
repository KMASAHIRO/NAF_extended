#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_doa_error_stats.py

- eval_circular_sweep_doa.py が保存した circular_eval_all.npz
  （err_deg を含む）と，
- 別スクリプトで保存した DoA誤差入りの .pkl

を入力として受け取り，
pred_vs_true_error（DoA誤差）について

  - 全サンプルの平均誤差 ± 標準偏差
  - 95パーセンタイルで外れ値除去したときの平均誤差 ± 標準偏差
  - 90パーセンタイルで外れ値除去したときの平均誤差 ± 標準偏差

を，それぞれのファイルごとに txt ファイルに出力する。

使い方の例:
    python compute_doa_error_stats.py \
        --npz path/to/a.npz path/to/b.npz \
        --pkl path/to/c.pkl path/to/d.pkl
"""

import os
import argparse
from typing import Optional
import pickle
import numpy as np


def load_errors_from_npz(path: str) -> np.ndarray:
    """circular_eval_all.npz から err_deg を取り出す"""
    data = np.load(path)
    if "err_deg" not in data:
        raise KeyError(f"'err_deg' not found in npz: {path}")
    err = np.asarray(data["err_deg"], dtype=np.float64)
    return err


def load_errors_from_pkl(path: str) -> np.ndarray:
    """
    pkl から pred_vs_true_error（DoA誤差）を取り出す。

    想定パターン:
      1) ルートが dict で 'pred_vs_true_error' キーを持つ
      2) ルートが dict で 'errors' キーの中に 'pred_vs_true_error' がある
      3) ルートがそのまま ndarray / list の場合（全体が誤差配列）
    必要に応じてここを調整してください。
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # パターン1: obj['pred_vs_true_error']
    if isinstance(obj, dict) and "pred_vs_true_error" in obj:
        arr = obj["pred_vs_true_error"]

    # パターン2: obj['errors']['pred_vs_true_error']
    elif isinstance(obj, dict) and "errors" in obj and isinstance(obj["errors"], dict) \
            and "pred_vs_true_error" in obj["errors"]:
        arr = obj["errors"]["pred_vs_true_error"]

    # パターン3: そのまま配列として扱う
    else:
        arr = obj

    err = np.asarray(arr, dtype=np.float64).reshape(-1)
    return err


def compute_stats(err: np.ndarray, percentile: Optional[float] = None):
    """
    誤差配列 err から mean, std を計算。
    percentile が指定されている場合は，
    そのパーセンタイル値以下のサンプルのみを用いる
    （上位 (100 - percentile)% を外れ値として除外）。
    """
    err = np.asarray(err, dtype=np.float64).reshape(-1)

    if percentile is not None:
        thr = float(np.percentile(err, percentile))
        mask = err <= thr  # これで上位側を除去（keep <= percentile）
        filtered = err[mask]
        if filtered.size == 0:
            return {
                "percentile": percentile,
                "threshold": thr,
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
            }
        return {
            "percentile": percentile,
            "threshold": thr,
            "n": int(filtered.size),
            "mean": float(filtered.mean()),
            "std": float(filtered.std(ddof=0)),
        }
    else:
        return {
            "percentile": None,
            "threshold": None,
            "n": int(err.size),
            "mean": float(err.mean()),
            "std": float(err.std(ddof=0)),
        }


def write_stats_txt(err: np.ndarray, out_txt_path: str, src_path: str):
    """誤差配列 err から統計量を計算し，テキストファイルに出力する。"""
    stats_all = compute_stats(err, percentile=None)
    stats_95 = compute_stats(err, percentile=95.0)
    stats_90 = compute_stats(err, percentile=90.0)

    lines = []
    lines.append(f"Source file: {src_path}")
    lines.append(f"N_total = {stats_all['n']}")
    lines.append("")
    lines.append("=== All samples ===")
    lines.append(f"mean_error_deg = {stats_all['mean']:.4f}")
    lines.append(f"std_error_deg  = {stats_all['std']:.4f}")
    lines.append("")
    lines.append("=== 95th percentile outlier removal (keep <= 95th) ===")
    lines.append(f"threshold_95th_deg = {stats_95['threshold']:.4f}")
    lines.append(f"N_used_95th        = {stats_95['n']}")
    lines.append(
        f"mean_error_deg_95  = {stats_95['mean']:.4f}"
        if stats_95['n'] > 0 else "mean_error_deg_95  = NaN"
    )
    lines.append(
        f"std_error_deg_95   = {stats_95['std']:.4f}"
        if stats_95['n'] > 0 else "std_error_deg_95   = NaN"
    )
    lines.append("")
    lines.append("=== 90th percentile outlier removal (keep <= 90th) ===")
    lines.append(f"threshold_90th_deg = {stats_90['threshold']:.4f}")
    lines.append(f"N_used_90th        = {stats_90['n']}")
    lines.append(
        f"mean_error_deg_90  = {stats_90['mean']:.4f}"
        if stats_90['n'] > 0 else "mean_error_deg_90  = NaN"
    )
    lines.append(
        f"std_error_deg_90   = {stats_90['std']:.4f}"
        if stats_90['n'] > 0 else "std_error_deg_90   = NaN"
    )
    lines.append("")

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] wrote stats: {out_txt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*", default=[], help="circular_eval_all.npz などの npz ファイル群")
    ap.add_argument("--pkl", nargs="*", default=[], help="pred_vs_true_error を含む pkl ファイル群")
    args = ap.parse_args()

    # npz
    for npz_path in args.npz:
        npz_path = os.path.abspath(npz_path)
        if not os.path.isfile(npz_path):
            print(f"[WARN] npz not found, skip: {npz_path}")
            continue
        try:
            err = load_errors_from_npz(npz_path)
        except Exception as e:
            print(f"[ERROR] failed to load npz {npz_path}: {e}")
            continue

        # 同じディレクトリに <stem>_doa_error_stats.txt を出力
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        out_txt = os.path.join(os.path.dirname(npz_path), f"{stem}_doa_error_stats.txt")
        write_stats_txt(err, out_txt, src_path=npz_path)

    # pkl
    for pkl_path in args.pkl:
        pkl_path = os.path.abspath(pkl_path)
        if not os.path.isfile(pkl_path):
            print(f"[WARN] pkl not found, skip: {pkl_path}")
            continue
        try:
            err = load_errors_from_pkl(pkl_path)
        except Exception as e:
            print(f"[ERROR] failed to load pkl {pkl_path}: {e}")
            continue

        stem = os.path.splitext(os.path.basename(pkl_path))[0]
        out_txt = os.path.join(os.path.dirname(pkl_path), f"{stem}_doa_error_stats.txt")
        write_stats_txt(err, out_txt, src_path=pkl_path)


if __name__ == "__main__":
    main()

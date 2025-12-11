#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAF DoA results.pkl 用 散布図 + 誤差統計作成スクリプト (DoA-B 専用)

対象:
- NAF DoA Evaluation Pipeline が出力した results.pkl
  (entries の各要素が {pred: {...}, ...} 形式, ori は存在しない前提)

出力:
1) 中央値ベース散布図 (1サブプロット)
   - x=true_deg, y=pred(median)      [DoA-B (median)]

2) 全点ベース散布図 (1サブプロット)
   - x=true_deg, y=pred(angles)      [DoA-B (all points)]

3) 誤差統計 (txt)
   - DoA-B について:
       mean ± std
       95パーセンタイル外れ値除去後の mean ± std
       90パーセンタイル外れ値除去後の mean ± std
"""

import os
import argparse
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

# ==========================================================
# ユーティリティ
# ==========================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def wrap_0_360(x: np.ndarray) -> np.ndarray:
    """角度を [0, 360) に wrap"""
    return np.mod(x, 360.0)


def angular_error_array(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """
    |a-b| を [0, 180] に短縮した角度差をベクトルで計算
    """
    a = np.asarray(a_deg, dtype=float)
    b = np.asarray(b_deg, dtype=float)
    diff = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def draw_subplot_single(ax, x, y, xlabel: str, ylabel: str, title: str):
    """
    汎用散布図 + x=y の赤点線
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return

    ax.scatter(x, y, alpha=0.5)
    ax.plot([0, 360], [0, 360], "r--")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, 361, 50))
    ax.set_yticks(np.arange(0, 361, 50))
    #ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)


# ==========================================================
# results.pkl ロード
# ==========================================================

def load_entries(pkl_path: str) -> List[Dict[str, Any]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("results.pkl の形式が想定と異なります: 'entries' が list ではありません。")
    return entries


# ==========================================================
# 中央値ベース散布用データ（DoA-B: true vs pred）
# ==========================================================

def collect_median_scatter_naf(entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    中央値ベースの散布用データ (DoA-B 用).

    戻り値:
      true_x             : x = true_deg
      pred_med_y_for_true: y = pred (中央値)
    """
    true_x = []
    pred_med_y_for_true = []

    for e in entries:
        pred = e.get("pred", {}) or {}

        true_deg = float(pred.get("true_deg", np.nan))

        # pred 側 angles
        pred_angles = np.asarray(pred.get("angles_deg", []), dtype=float)
        if pred_angles.size == 0:
            continue

        pred_med = float(np.median(pred_angles))
        pred_med = float(pred_med % 360.0)

        # true vs pred (中央値) [DoA-B]
        if np.isfinite(true_deg) and np.isfinite(pred_med):
            true_x.append(float(true_deg % 360.0))
            pred_med_y_for_true.append(pred_med)

    return (
        np.asarray(true_x, dtype=float),
        np.asarray(pred_med_y_for_true, dtype=float),
    )


# ==========================================================
# 全点ベース散布用データ（DoA-B: true vs pred）
# ==========================================================

def collect_allpoints_scatter_naf(entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    全点（窓ごとの angles_deg）ベースの散布用データ (DoA-B 用).

    戻り値:
      true_all_x          : x = true_deg(各窓)
      pred_all_y_for_true : y = pred(各窓)
    """
    true_all_x = []
    pred_all_y_for_true = []

    for e in entries:
        pred = e.get("pred", {}) or {}

        true_deg = float(pred.get("true_deg", np.nan))
        pred_angles = np.asarray(pred.get("angles_deg", []), dtype=float)

        # true vs pred (全点) [DoA-B]
        if np.isfinite(true_deg) and pred_angles.size > 0:
            true_val = float(true_deg % 360.0)
            true_all_x.extend([true_val] * pred_angles.size)
            pred_all_y_for_true.extend(wrap_0_360(pred_angles).tolist())

    return (
        np.asarray(true_all_x, dtype=float),
        np.asarray(pred_all_y_for_true, dtype=float),
    )


# ==========================================================
# プロット関数（DoA-B のみ）
# ==========================================================

def plot_median_scatter(true_x, pred_med_y_for_true, outdir: str):
    """
    中央値ベースの散布図を 1 サブプロットで描画:
      - x=true_deg, y=pred(median)  [DoA-B median]
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    draw_subplot_single(
        ax,
        true_x,
        pred_med_y_for_true,
        xlabel="真の音源方向 [°]",
        ylabel="予測定位方向 [°]",
        title="DoA-B (median): Pred vs true",
    )

    plt.tight_layout()
    out_path = os.path.join(outdir, "scatter_median_true_vs_pred.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_allpoints_scatter(true_all_x, pred_all_y_for_true, outdir: str):
    """
    全点ベースの散布図を 1 サブプロットで描画:
      - x=true_deg, y=pred(angles) [DoA-B all points]
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    draw_subplot_single(
        ax,
        true_all_x,
        pred_all_y_for_true,
        xlabel="true angle [deg]",
        ylabel="pred angle (per window) [deg]",
        title="DoA-B (all points): Pred vs true",
    )

    plt.tight_layout()
    out_path = os.path.join(outdir, "scatter_allpoints_true_vs_pred.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[SAVE] {out_path}")


# ==========================================================
# 誤差統計 (txt 出力, DoA-B のみ)
# ==========================================================

def write_error_stats(err: np.ndarray, label: str, f):
    """
    err: 誤差配列（deg, [0,180]）
    label: 出力セクション名
    f: 書き込み用ファイルオブジェクト
    """
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]

    f.write(f"=== {label} ===\n")
    if err.size == 0:
        f.write("num_samples: 0\n")
        f.write("mean±std: nan ± nan\n")
        f.write("95p (thr): nan, after removal: mean±std = nan ± nan, n=0\n")
        f.write("90p (thr): nan, after removal: mean±std = nan ± nan, n=0\n\n")
        return

    # full
    mean_full = float(err.mean())
    std_full = float(err.std(ddof=1)) if err.size > 1 else float("nan")

    # 95 percentile
    thr95 = float(np.percentile(err, 95))
    kept95 = err[err <= thr95]
    mean95 = float(kept95.mean()) if kept95.size > 0 else float("nan")
    std95 = float(kept95.std(ddof=1)) if kept95.size > 1 else float("nan")

    # 90 percentile
    thr90 = float(np.percentile(err, 90))
    kept90 = err[err <= thr90]
    mean90 = float(kept90.mean()) if kept90.size > 0 else float("nan")
    std90 = float(kept90.std(ddof=1)) if kept90.size > 1 else float("nan")

    f.write(f"num_samples: {err.size}\n")
    f.write(f"mean±std: {mean_full:.4f} ± {std_full:.4f}\n")
    f.write(
        f"95p (thr={thr95:.4f}): mean±std = {mean95:.4f} ± {std95:.4f}, n={kept95.size}\n"
    )
    f.write(
        f"90p (thr={thr90:.4f}): mean±std = {mean90:.4f} ± {std90:.4f}, n={kept90.size}\n"
    )
    f.write("\n")


def save_all_error_stats(
    true_x, pred_med_y_for_true,
    true_all_x, pred_all_y_for_true,
    outdir: str,
):
    """
    DoA-B の誤差統計を error_stats.txt に書き出す。
    - median レベル
    - all points レベル
    """
    _ensure_dir(outdir)
    txt_path = os.path.join(outdir, "error_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        # --- median level (DoA-B) ---
        err_med_true = angular_error_array(pred_med_y_for_true, true_x)
        write_error_stats(err_med_true, "DoA-B (median): true vs pred(median)", f)

        # --- all points level (DoA-B) ---
        err_all_true = angular_error_array(pred_all_y_for_true, true_all_x)
        write_error_stats(err_all_true, "DoA-B (all points): true vs pred(angles)", f)

    print(f"[SAVE] {txt_path}")


# ==========================================================
# main
# ==========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True, help="Path to NAF results.pkl")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for figures & stats")
    args = ap.parse_args()

    pkl_path = os.path.expanduser(args.pkl)
    outdir = os.path.expanduser(args.outdir)

    entries = load_entries(pkl_path)
    print(f"[INFO] Loaded {len(entries)} entries from {pkl_path}")

    # ---- 中央値ベース (DoA-B) ----
    true_x, pred_med_y_for_true = collect_median_scatter_naf(entries)
    print(f"[INFO] median-level pairs (true vs pred): {true_x.size}")
    plot_median_scatter(true_x, pred_med_y_for_true, outdir)

    # ---- 全点ベース (DoA-B) ----
    true_all_x, pred_all_y_for_true = collect_allpoints_scatter_naf(entries)
    print(f"[INFO] all-point pairs (true vs pred): {true_all_x.size}")
    plot_allpoints_scatter(true_all_x, pred_all_y_for_true, outdir)

    # ---- 誤差統計 (txt, DoA-B) ----
    save_all_error_stats(
        true_x, pred_med_y_for_true,
        true_all_x, pred_all_y_for_true,
        outdir,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAF DoA results.pkl 用 散布図 + 誤差統計作成スクリプト

対象:
- NAF DoA Evaluation Pipeline が出力した results.pkl
  (entries の各要素が {pred: {...}, ori: {...}, ...} 形式)

出力:
1) 中央値ベース散布図 (2サブプロット)
   - x=gt(ori median), y=pred(median)      [DoA-A (median)]
   - x=true_deg,       y=pred(median)      [DoA-B (median)]

2) 全点ベース散布図 (2サブプロット)
   - x=gt(ori angles), y=pred(angles)      [DoA-A (all points)]
   - x=true_deg,       y=pred(angles)      [DoA-B (all points)]

3) 誤差統計 (txt)
   - 各ペアについて:
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
# 中央値ベース散布用データ
# ==========================================================

def collect_median_scatter_naf(entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    中央値ベースの散布用データを作成する (NAF 用).

    戻り値:
      gt_med_x, pred_med_y_for_gt, true_x, pred_med_y_for_true
    - gt_med_x           : x = gt (ori の中央値)
    - pred_med_y_for_gt  : y = pred (中央値) [gt vs pred 用]
    - true_x             : x = true_deg
    - pred_med_y_for_true: y = pred (中央値) [true vs pred 用]
    """
    gt_med_x = []
    pred_med_y_for_gt = []
    true_x = []
    pred_med_y_for_true = []

    for e in entries:
        pred = e.get("pred", {}) or {}
        ori = e.get("ori", None)

        true_deg = float(pred.get("true_deg", np.nan))

        # pred 側 angles
        pred_angles = np.asarray(pred.get("angles_deg", []), dtype=float)
        if pred_angles.size == 0:
            continue
        pred_med = float(np.median(pred_angles))
        pred_med = float(pred_med % 360.0)

        # gt 側 angles (ori)
        if ori is not None:
            gt_angles = np.asarray(ori.get("angles_deg", []), dtype=float)
            if gt_angles.size > 0:
                gt_med = float(np.median(gt_angles))
                gt_med = float(gt_med % 360.0)
                if np.isfinite(gt_med) and np.isfinite(pred_med):
                    gt_med_x.append(gt_med)
                    pred_med_y_for_gt.append(pred_med)

        # true vs pred (中央値)
        if np.isfinite(true_deg) and np.isfinite(pred_med):
            true_x.append(float(true_deg % 360.0))
            pred_med_y_for_true.append(pred_med)

    return (
        np.asarray(gt_med_x, dtype=float),
        np.asarray(pred_med_y_for_gt, dtype=float),
        np.asarray(true_x, dtype=float),
        np.asarray(pred_med_y_for_true, dtype=float),
    )


# ==========================================================
# 全点ベース散布用データ
# ==========================================================

def collect_allpoints_scatter_naf(entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    全点（窓ごとの angles_deg）ベースの散布用データを作成する (NAF 用).

    戻り値:
      gt_all_x, pred_all_y_for_gt, true_all_x, pred_all_y_for_true
    - gt_all_x            : x = gt(ori 各窓)
    - pred_all_y_for_gt   : y = pred(各窓)   [gt vs pred 用]
    - true_all_x          : x = true_deg(各窓)
    - pred_all_y_for_true : y = pred(各窓)   [true vs pred 用]
    """
    gt_all_x = []
    pred_all_y_for_gt = []
    true_all_x = []
    pred_all_y_for_true = []

    for e in entries:
        pred = e.get("pred", {}) or {}
        ori = e.get("ori", None)

        true_deg = float(pred.get("true_deg", np.nan))
        pred_angles = np.asarray(pred.get("angles_deg", []), dtype=float)

        # gt vs pred (全点): ori 側がある場合のみ
        if ori is not None:
            gt_angles = np.asarray(ori.get("angles_deg", []), dtype=float)
            n_pair = min(pred_angles.size, gt_angles.size)
            if n_pair > 0:
                gt_all_x.extend(wrap_0_360(gt_angles[:n_pair]).tolist())
                pred_all_y_for_gt.extend(wrap_0_360(pred_angles[:n_pair]).tolist())

        # true vs pred (全点)
        if np.isfinite(true_deg) and pred_angles.size > 0:
            true_val = float(true_deg % 360.0)
            true_all_x.extend([true_val] * pred_angles.size)
            pred_all_y_for_true.extend(wrap_0_360(pred_angles).tolist())

    return (
        np.asarray(gt_all_x, dtype=float),
        np.asarray(pred_all_y_for_gt, dtype=float),
        np.asarray(true_all_x, dtype=float),
        np.asarray(pred_all_y_for_true, dtype=float),
    )


# ==========================================================
# プロット関数
# ==========================================================

def plot_median_scatter(gt_med_x, pred_med_y_for_gt, true_x, pred_med_y_for_true, outdir: str):
    """
    中央値ベースの散布図を 2 サブプロットで描画:
      - 左: x=gt(ori median), y=pred(median)  [DoA-A median]
      - 右: x=true_deg,       y=pred(median)  [DoA-B median]
    """
    _ensure_dir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # gt vs pred (median)
    draw_subplot_single(
        axes[0],
        gt_med_x,
        pred_med_y_for_gt,
        xlabel="正解定位方向 [°]",
        ylabel="予測定位方向 [°]",
        title="DoA-A (median): Pred vs gt (ori median)",
    )

    # true vs pred (median)
    draw_subplot_single(
        axes[1],
        true_x,
        pred_med_y_for_true,
        xlabel="真の音源方向 [°]",
        ylabel="予測定位方向 [°]",
        title="DoA-B (median): Pred vs true",
    )

    plt.tight_layout()
    out_path = os.path.join(outdir, "scatter_median_gt_true_vs_pred.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_allpoints_scatter(gt_all_x, pred_all_y_for_gt, true_all_x, pred_all_y_for_true, outdir: str):
    """
    全点ベースの散布図を 2 サブプロットで描画:
      - 左: x=gt(ori angles), y=pred(angles) [DoA-A all points]
      - 右: x=true_deg,       y=pred(angles) [DoA-B all points]
    """
    _ensure_dir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # gt vs pred (all points)
    draw_subplot_single(
        axes[0],
        gt_all_x,
        pred_all_y_for_gt,
        xlabel="gt angle (ori per window) [deg]",
        ylabel="pred angle (per window) [deg]",
        title="DoA-A (all points): Pred vs gt (ori)",
    )

    # true vs pred (all points)
    draw_subplot_single(
        axes[1],
        true_all_x,
        pred_all_y_for_true,
        xlabel="true angle [deg]",
        ylabel="pred angle (per window) [deg]",
        title="DoA-B (all points): Pred vs true",
    )

    plt.tight_layout()
    out_path = os.path.join(outdir, "scatter_allpoints_gt_true_vs_pred.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[SAVE] {out_path}")


# ==========================================================
# 誤差統計 (txt 出力)
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
    gt_med_x, pred_med_y_for_gt,
    true_x, pred_med_y_for_true,
    gt_all_x, pred_all_y_for_gt,
    true_all_x, pred_all_y_for_true,
    outdir: str,
):
    """
    4パターン分の誤差統計を error_stats.txt に書き出す。
    """
    _ensure_dir(outdir)
    txt_path = os.path.join(outdir, "error_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        # --- median level ---
        err_med_gt   = angular_error_array(pred_med_y_for_gt, gt_med_x)
        err_med_true = angular_error_array(pred_med_y_for_true, true_x)

        write_error_stats(err_med_gt,   "DoA-A (median): gt(ori median) vs pred(median)", f)
        write_error_stats(err_med_true, "DoA-B (median): true vs pred(median)",          f)

        # --- all points level ---
        err_all_gt   = angular_error_array(pred_all_y_for_gt, gt_all_x)
        err_all_true = angular_error_array(pred_all_y_for_true, true_all_x)

        write_error_stats(err_all_gt,   "DoA-A (all points): gt(ori angles) vs pred(angles)", f)
        write_error_stats(err_all_true, "DoA-B (all points): true vs pred(angles)",           f)

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

    # ---- 中央値ベース ----
    gt_med_x, pred_med_y_for_gt, true_x, pred_med_y_for_true = collect_median_scatter_naf(entries)
    print(f"[INFO] median-level pairs: gt={gt_med_x.size}, true={true_x.size}")
    plot_median_scatter(gt_med_x, pred_med_y_for_gt, true_x, pred_med_y_for_true, outdir)

    # ---- 全点ベース ----
    gt_all_x, pred_all_y_for_gt, true_all_x, pred_all_y_for_true = collect_allpoints_scatter_naf(entries)
    print(f"[INFO] all-point pairs: gt={gt_all_x.size}, true={true_all_x.size}")
    plot_allpoints_scatter(gt_all_x, pred_all_y_for_gt, true_all_x, pred_all_y_for_true, outdir)

    # ---- 誤差統計 (txt) ----
    save_all_error_stats(
        gt_med_x, pred_med_y_for_gt,
        true_x,   pred_med_y_for_true,
        gt_all_x, pred_all_y_for_gt,
        true_all_x, pred_all_y_for_true,
        outdir,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whitenoise_eval_suite.py (summary_whitenoise.csv に誤差の平均と標準偏差を出力)

- A) 角度グリッド: angles_pred_ori_grid.png
- B) フレーム散布 & 統計
- C) 波形(=entry)単位の評価（mu/med使用）
- D) ルート直下に summary_whitenoise.csv を1つ出力（各pkl=1行）
"""

import os
import pickle
import csv
from glob import glob
from typing import Dict, Any, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ====================== 基本ユーティリティ ======================

def wrap_signed(d: np.ndarray) -> np.ndarray:
    """角度差(度)を [-180, 180) にラップ"""
    return (d + 180.0) % 360.0 - 180.0

def mod360(a: np.ndarray) -> np.ndarray:
    return a % 360.0

def to_array(x, dtype=float) -> np.ndarray:
    if x is None:
        return np.array([], dtype=dtype)
    return np.asarray(x, dtype=dtype)

def _title_for_subplot(e_pred: Dict[str, Any]) -> str:
    g = e_pred.get('index', e_pred.get('group', '?'))
    try:
        gi = int(g)
    except Exception:
        gi = -1
    td = float(e_pred.get('true_deg', float('nan')))
    return f"G{gi:02d} (true={td:.1f}°)"

def mean_or_nan(a: np.ndarray) -> float:
    return float(np.nanmean(a)) if a.size > 0 else float('nan')

def std_or_nan(a: np.ndarray) -> float:
    return float(np.nanstd(a)) if a.size > 0 else float('nan')


# ====================== 取り出し（pkl構造に準拠） ======================

def _angles_centers(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    ang = to_array(d.get("angles_deg", []), float)
    cen = to_array(d.get("centers", []), int)
    return mod360(ang), cen

def extract_entry(e: Dict[str, Any]):
    """
    entries[i] から pred/ori/true と計算済み代表角を取り出す
    戻り値:
      pred_angles, pred_centers, ori_angles, ori_centers,
      true_deg, pred_mu, pred_med, ori_mu, ori_med
    """
    pred = e.get("pred", {}) or {}
    ori  = e.get("ori",  {}) or {}

    pred_angles, pred_centers = _angles_centers(pred)
    ori_angles,  ori_centers  = _angles_centers(ori)

    # true は pred に入っている前提（なければ ori 側を参照）
    if "true_deg" in pred:
        true_deg = float(pred["true_deg"])
    elif "true_deg" in ori:
        true_deg = float(ori["true_deg"])
    else:
        true_deg = float('nan')

    def _get(dct: Dict[str, Any], k: str) -> float:
        v = dct.get(k, float('nan'))
        return float(v) if v is not None else float('nan')

    pred_mu  = _get(pred, "mu_deg")
    pred_med = _get(pred, "med_deg")
    ori_mu   = _get(ori,  "mu_deg")
    ori_med  = _get(ori,  "med_deg")

    return (pred_angles, pred_centers, ori_angles, ori_centers,
            true_deg, pred_mu, pred_med, ori_mu, ori_med)


# ====================== A) 角度グリッド ======================

def plot_angles_grid(pkl_path: str, rows: int = 3, cols: int = 6,
                     ylim: Tuple[float, float] = (0.0, 360.0), dpi: int = 250) -> None:
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {pkl_path}: {e}")
        return

    entries = data.get("entries", [])
    if not entries:
        print(f"[SKIP] no entries in {pkl_path}")
        return

    R, C = rows, cols
    fig, axes = plt.subplots(R, C, figsize=(4.6*C, 2.9*R), sharex=False, sharey=True)
    axes = axes.ravel()
    last_ax_idx = -1

    for i, e in enumerate(entries):
        if i >= R*C: break
        last_ax_idx = i
        ax = axes[i]

        (pred_angles, pred_centers, ori_angles, ori_centers, true_deg,
         _, _, _, _) = extract_entry(e)

        has_pred = (pred_angles.size > 0 and pred_centers.size > 0)
        has_ori  = (ori_angles.size  > 0 and ori_centers.size  > 0)

        if not has_pred and not has_ori:
            ax.set_title(_title_for_subplot(e.get("pred", {})) + "\n[pred/ori: no frames]", fontsize=9)
            if np.isfinite(true_deg):
                ax.axhline(true_deg % 360.0, color="red", lw=1.0, alpha=0.9, linestyle='--')
            ax.set_ylim(*ylim); ax.grid(True, alpha=0.3)
            ax.set_xlabel("Window center (STFT frame)")
            if i % C == 0: ax.set_ylabel("Angle [deg]")
            continue

        if has_pred:
            ax.scatter(pred_centers, mod360(pred_angles), s=6, alpha=0.65, label="pred")
        else:
            ax.text(0.02, 0.85, "pred: n/a", transform=ax.transAxes, fontsize=8)

        if has_ori:
            ax.scatter(ori_centers, mod360(ori_angles), s=8, marker='^', alpha=0.7, label="ori")
        else:
            ax.text(0.02, 0.75, "ori: n/a", transform=ax.transAxes, fontsize=8)

        if np.isfinite(true_deg):
            ax.axhline(true_deg % 360.0, color="red", lw=1.2, alpha=0.9)

        ax.set_title(_title_for_subplot(e.get("pred", {})), fontsize=9)
        ax.set_ylim(*ylim); ax.grid(True, alpha=0.3)
        ax.set_xlabel("Window center (STFT frame)")
        if i % C == 0: ax.set_ylabel("Angle [deg]")

        if i == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.8)

    for j in range(last_ax_idx + 1, R * C):
        fig.delaxes(axes[j])

    meta = data.get("meta", {})
    cond = meta.get("condition", "")
    stft = f"L{meta.get('stft_nfft','?')}_H{meta.get('stft_hop','?')}_{meta.get('stft_win','?')}"
    Tuse = meta.get("Tuse", None)
    fig.suptitle(f"{cond} | STFT {stft}" + (f" | T_use={Tuse}" if Tuse is not None else "") +
                 "  (angles: pred & ori)", fontsize=12, y=1.02)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(pkl_path), "angles_pred_ori_grid.png")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("[OK] saved:", out_png)


# ====================== B) フレーム散布 & 統計 ======================

def collect_scatter_points(entries: Sequence[dict]):
    xs_true_ori, ys_true_ori = [], []
    xs_true_pred, ys_true_pred = [], []
    xs_ori_pred, ys_ori_pred = [], []

    for e in entries:
        (pred_angles, pred_centers, ori_angles, ori_centers, true_deg,
         _, _, _, _) = extract_entry(e)

        if ori_angles.size > 0 and np.isfinite(true_deg):
            xs_true_ori.append(np.full_like(ori_angles, fill_value=true_deg, dtype=float))
            ys_true_ori.append(ori_angles)

        if pred_angles.size > 0 and np.isfinite(true_deg):
            xs_true_pred.append(np.full_like(pred_angles, fill_value=true_deg, dtype=float))
            ys_true_pred.append(pred_angles)

        if (ori_angles.size > 0) and (pred_angles.size > 0):
            pred_map = {int(f): ang for f, ang in zip(pred_centers.tolist(), pred_angles.tolist())}
            ori_map  = {int(f): ang for f, ang in zip(ori_centers.tolist(),  ori_angles.tolist())}
            common = sorted(set(pred_map.keys()) & set(ori_map.keys()))
            if common:
                xs_ori_pred.append(np.array([ori_map[f]  for f in common], dtype=float))
                ys_ori_pred.append(np.array([pred_map[f] for f in common], dtype=float))

    def _cat(L):
        if not L: return np.array([], dtype=float)
        return np.concatenate(L)

    return (_cat(xs_true_ori), _cat(ys_true_ori),
            _cat(xs_true_pred), _cat(ys_true_pred),
            _cat(xs_ori_pred),  _cat(ys_ori_pred))

def collect_frame_errors(entries: Sequence[dict]):
    """フレーム単位の誤差（絶対値）"""
    e_ori_true, e_pred_true, e_pred_ori = [], [], []
    for e in entries:
        (pred_angles, pred_centers, ori_angles, ori_centers, true_deg,
         _, _, _, _) = extract_entry(e)

        if ori_angles.size > 0 and np.isfinite(true_deg):
            e_ori_true.append(np.abs(wrap_signed(ori_angles - true_deg)))
        if pred_angles.size > 0 and np.isfinite(true_deg):
            e_pred_true.append(np.abs(wrap_signed(pred_angles - true_deg)))
        if (ori_angles.size > 0) and (pred_angles.size > 0):
            pred_map = {int(f): ang for f, ang in zip(pred_centers.tolist(), pred_angles.tolist())}
            ori_map  = {int(f): ang for f, ang in zip(ori_centers.tolist(),  ori_angles.tolist())}
            common = sorted(set(pred_map.keys()) & set(ori_map.keys()))
            if common:
                pa = np.array([pred_map[f] for f in common], dtype=float)
                oa = np.array([ori_map[f]  for f in common], dtype=float)
                e_pred_ori.append(np.abs(wrap_signed(pa - oa)))

    def _cat(L):
        if not L: return np.array([], dtype=float)
        return np.concatenate(L)

    return _cat(e_ori_true), _cat(e_pred_true), _cat(e_pred_ori)

def plot_scatter_all(out_png: str,
                     xs_true_ori: np.ndarray, ys_true_ori: np.ndarray,
                     xs_true_pred: np.ndarray, ys_true_pred: np.ndarray,
                     xs_ori_pred: np.ndarray,  ys_ori_pred: np.ndarray):
    if xs_true_ori.size==0 and xs_true_pred.size==0 and xs_ori_pred.size==0:
        print(f"[SKIP] no scatter data -> {out_png}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.ravel()

    def sp(ax, x, y, xl, yl, ttl):
        if x.size == 0 or y.size == 0:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=12, color='gray')
        else:
            ax.scatter(mod360(x), mod360(y), s=10, alpha=0.6)
        ax.set_xlim(0, 360); ax.set_ylim(0, 360)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')

    sp(axes[0], xs_true_ori,  ys_true_ori,  "true [deg]", "ori [deg]",   "ori vs true (frames)")
    sp(axes[1], xs_true_pred, ys_true_pred, "true [deg]", "pred [deg]",  "pred vs true (frames)")
    sp(axes[2], xs_ori_pred,  ys_ori_pred, "ori [deg]",  "pred [deg]",  "pred vs ori (common frames)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches='tight')
    plt.close()
    print("[OK] saved:", out_png)


# ====================== C) 波形(=entry)単位（既計算のmu/medを使用） ======================

def collect_wave_level(entries: Sequence[dict], which: str):
    """
    which: 'mean' -> mu_deg を代表角に, 'median' -> med_deg を代表角に
    戻り:
      true_list, ori_rep_list, pred_rep_list,
      abs_ori_true, abs_pred_true, abs_pred_ori
    """
    trues, ori_rep, pred_rep = [], [], []
    e_ori_true, e_pred_true, e_pred_ori = [], [], []

    for e in entries:
        ( _, _, _, _, true_deg,
          pred_mu, pred_med, ori_mu, ori_med) = extract_entry(e)

        pred_r = pred_mu if which == "mean" else pred_med
        ori_r  = ori_mu  if which == "mean" else ori_med

        trues.append(true_deg)
        ori_rep.append(ori_r)
        pred_rep.append(pred_r)

        if np.isfinite(ori_r) and np.isfinite(true_deg):
            e_ori_true.append(abs(wrap_signed(np.array([ori_r - true_deg]))))
        if np.isfinite(pred_r) and np.isfinite(true_deg):
            e_pred_true.append(abs(wrap_signed(np.array([pred_r - true_deg]))))
        if np.isfinite(pred_r) and np.isfinite(ori_r):
            e_pred_ori.append(abs(wrap_signed(np.array([pred_r - ori_r]))))

    def _cat(L):
        if not L: return np.array([], dtype=float)
        return np.concatenate(L)

    return (np.asarray(trues, float),
            np.asarray(ori_rep, float),
            np.asarray(pred_rep, float),
            _cat(e_ori_true), _cat(e_pred_true), _cat(e_pred_ori))

def plot_wave_scatter(out_png: str,
                      true_list: np.ndarray,
                      ori_rep: np.ndarray,
                      pred_rep: np.ndarray,
                      title_suffix: str):
    if true_list.size==0 and ori_rep.size==0 and pred_rep.size==0:
        print(f"[SKIP] no waveform scatter data -> {out_png}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.ravel()

    def sp(ax, x, y, xl, yl, ttl):
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) == 0:
            ax.text(0.5, 0.5, "no data", ha='center', va='center', fontsize=12, color='gray')
        else:
            ax.scatter(mod360(x[mask]), mod360(y[mask]), s=25, alpha=0.7)
        ax.set_xlim(0, 360); ax.set_ylim(0, 360)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')

    sp(axes[0], true_list, ori_rep,  "true [deg]", "ori rep. [deg]",  f"ori vs true ({title_suffix})")
    sp(axes[1], true_list, pred_rep, "true [deg]", "pred rep. [deg]", f"pred vs true ({title_suffix})")
    sp(axes[2], ori_rep,   pred_rep, "ori rep. [deg]", "pred rep. [deg]", f"pred vs ori ({title_suffix})")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches='tight')
    plt.close()
    print("[OK] saved:", out_png)


# ====================== メイン ======================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="pkl探索ルート")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--ang_ylim", type=float, nargs=2, default=[0, 360])

    # 出力ON/OFF
    ap.add_argument("--no_grids", action="store_true", help="角度グリッド（angles_pred_ori_grid.png）を出力しない")
    ap.add_argument("--no_scatter_frames", action="store_true")
    ap.add_argument("--no_wave_plots", action="store_true")

    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    pkl_files = sorted(glob(os.path.join(root, "**", "results.pkl"), recursive=True))
    if not pkl_files:
        print("[ERROR] No results.pkl found under", root)
        return

    print(f"[INFO] Found {len(pkl_files)} results.pkl files")

    # CSV 収集バッファ
    csv_header = [
        "exp_dir_relative",
        # Frame-level
        "frame_abs_ori_true_mean_deg",  "frame_abs_ori_true_std_deg",
        "frame_abs_pred_true_mean_deg", "frame_abs_pred_true_std_deg",
        "frame_abs_pred_ori_mean_deg",  "frame_abs_pred_ori_std_deg",
        # Wave-level (mu)
        "wave_abs_ori_mu_true_mean_deg",  "wave_abs_ori_mu_true_std_deg",
        "wave_abs_pred_mu_true_mean_deg", "wave_abs_pred_mu_true_std_deg",
        "wave_abs_pred_mu_ori_mu_mean_deg","wave_abs_pred_mu_ori_mu_std_deg",
        # Wave-level (median)
        "wave_abs_ori_med_true_mean_deg",  "wave_abs_ori_med_true_std_deg",
        "wave_abs_pred_med_true_mean_deg", "wave_abs_pred_med_true_std_deg",
        "wave_abs_pred_med_ori_med_mean_deg","wave_abs_pred_med_ori_med_std_deg",
    ]
    summary_rows = []

    for pkl_path in tqdm(pkl_files, desc="EvalSuite"):
        # A) 角度グリッド
        if not args.no_grids:
            plot_angles_grid(pkl_path, rows=args.rows, cols=args.cols,
                             ylim=tuple(args.ang_ylim), dpi=args.dpi)

        # B) フレームスキャッタ & 統計
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {pkl_path}: {e}")
            continue

        entries = data.get("entries", [])

        (xs_true_ori, ys_true_ori,
         xs_true_pred, ys_true_pred,
         xs_ori_pred,  ys_ori_pred) = collect_scatter_points(entries)

        e_ori_true, e_pred_true, e_pred_ori = collect_frame_errors(entries)
        m_ori_true,  s_ori_true  = mean_or_nan(e_ori_true),  std_or_nan(e_ori_true)
        m_pred_true, s_pred_true = mean_or_nan(e_pred_true), std_or_nan(e_pred_true)
        m_pred_ori,  s_pred_ori  = mean_or_nan(e_pred_ori),  std_or_nan(e_pred_ori)

        exp_dir = os.path.dirname(pkl_path)
        print(
            f"[FRAME-LEVEL STATS] {exp_dir}\n"
            f"  ori-true  mean±std = {m_ori_true:.3f} ± {s_ori_true:.3f} deg  (frames)\n"
            f"  pred-true mean±std = {m_pred_true:.3f} ± {s_pred_true:.3f} deg  (frames)\n"
            f"  pred-ori  mean±std = {m_pred_ori:.3f} ± {s_pred_ori:.3f} deg  (common frames)\n"
        )

        if not args.no_scatter_frames:
            out_scatter = os.path.join(exp_dir, "scatter_all.png")
            plot_scatter_all(out_scatter,
                             xs_true_ori, ys_true_ori,
                             xs_true_pred, ys_true_pred,
                             xs_ori_pred,  ys_ori_pred)

        # C) 波形（mu）
        (true_mu, ori_mu, pred_mu,
         abs_ori_true_mu, abs_pred_true_mu, abs_pred_ori_mu) = collect_wave_level(entries, which="mean")

        m1, s1 = mean_or_nan(abs_ori_true_mu),  std_or_nan(abs_ori_true_mu)
        m2, s2 = mean_or_nan(abs_pred_true_mu), std_or_nan(abs_pred_true_mu)
        m3, s3 = mean_or_nan(abs_pred_ori_mu),  std_or_nan(abs_pred_ori_mu)

        print(
            f"[WAVE-LEVEL STATS / MEAN] {exp_dir}\n"
            f"  ori(mu)-true      mean±std = {m1:.3f} ± {s1:.3f} deg\n"
            f"  pred(mu)-true     mean±std = {m2:.3f} ± {s2:.3f} deg\n"
            f"  pred(mu)-ori(mu)  mean±std = {m3:.3f} ± {s3:.3f} deg\n"
        )

        if not args.no_wave_plots:
            out_wave = os.path.join(exp_dir, "scatter_wave_all.png")
            plot_wave_scatter(out_wave, true_mu, ori_mu, pred_mu, title_suffix="mean over time")

        # C') 波形（median）
        (true_md, ori_md, pred_md,
         abs_ori_true_md, abs_pred_true_md, abs_pred_ori_md) = collect_wave_level(entries, which="median")

        n1, t1 = mean_or_nan(abs_ori_true_md),  std_or_nan(abs_ori_true_md)
        n2, t2 = mean_or_nan(abs_pred_true_md), std_or_nan(abs_pred_true_md)
        n3, t3 = mean_or_nan(abs_pred_ori_md),  std_or_nan(abs_pred_ori_md)

        print(
            f"[WAVE-LEVEL STATS / MEDIAN] {exp_dir}\n"
            f"  ori(med)-true      mean±std = {n1:.3f} ± {t1:.3f} deg\n"
            f"  pred(med)-true     mean±std = {n2:.3f} ± {t2:.3f} deg\n"
            f"  pred(med)-ori(med) mean±std = {n3:.3f} ± {t3:.3f} deg\n"
        )

        if not args.no_wave_plots:
            out_wave_med = os.path.join(exp_dir, "scatter_wave_median_all.png")
            plot_wave_scatter(out_wave_med, true_md, ori_md, pred_md, title_suffix="median over time")

        # CSV 行を追加
        rel_dir = os.path.relpath(exp_dir, root)
        summary_rows.append([
            rel_dir,
            f"{m_ori_true:.6f}",  f"{s_ori_true:.6f}",
            f"{m_pred_true:.6f}", f"{s_pred_true:.6f}",
            f"{m_pred_ori:.6f}",  f"{s_pred_ori:.6f}",
            f"{m1:.6f}", f"{s1:.6f}",
            f"{m2:.6f}", f"{s2:.6f}",
            f"{m3:.6f}", f"{s3:.6f}",
            f"{n1:.6f}", f"{t1:.6f}",
            f"{n2:.6f}", f"{t2:.6f}",
            f"{n3:.6f}", f"{t3:.6f}",
        ])

    # CSV 書き出し（--root 直下）
    out_csv = os.path.join(root, "summary_whitenoise.csv")
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(summary_rows)
        print(f"[OK] summary CSV saved: {out_csv}")
    except Exception as e:
        print(f"[WARN] Failed to write summary CSV: {e}")


if __name__ == "__main__":
    main()

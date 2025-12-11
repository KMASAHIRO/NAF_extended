#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AVR & NAF 共通プロッタ
-----------------------------
横軸: trial index
縦軸: DoA-A (DoA error, [deg])
"""

import os
import glob
import pickle
import argparse
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import optuna


# ===================== AVR側ユーティリティ =====================

def _load_mean_error_from_pkl(
    pkl_path: str,
    algo_name: str,
    error_key: str,
) -> Optional[Tuple[float, int]]:
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        errs = data.get(algo_name, {}).get(error_key, [])
        errs = [e for e in errs if e is not None]
        if len(errs) == 0:
            return None

        err_arr = np.asarray(errs, dtype=float)
        mean_err = float(np.mean(err_arr))

        base = os.path.basename(pkl_path)
        name, _ = os.path.splitext(base)
        try:
            it = int(name.replace("val_iter", ""))
        except Exception:
            it = -1

        return mean_err, it
    except Exception:
        return None


def collect_avr_doa_curve_for_logdir(
    logdir: str,
    trial_names: List[str],
    algo_name: str = "NormMUSIC",
    error_key: str = "pred_vs_gt_error",
    skip_missing: bool = False,
) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []

    idx = 1
    for trial_name in trial_names:
        doa_dir = os.path.join(logdir, trial_name, "doa_results")
        pkl_files = sorted(glob.glob(os.path.join(doa_dir, "val_iter*.pkl")))
        best_mean = None

        for p in pkl_files:
            res = _load_mean_error_from_pkl(p, algo_name, error_key)
            if res is None:
                continue
            mean_err, it = res
            if best_mean is None or mean_err < best_mean:
                best_mean = mean_err

        if best_mean is None:
            if skip_missing:
                continue
            else:
                xs.append(idx)
                ys.append(np.nan)
                idx += 1
                continue

        xs.append(idx)
        ys.append(best_mean)
        idx += 1

    return xs, ys


# ===================== NAF側ユーティリティ =====================

def collect_naf_doa_curve_from_optuna(
    storage: str,
    study_name: str,
    threshold: float = 90.0,
    max_trials: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [t for t in study.trials if t.value is not None]

    if not trials:
        print(f"[WARN] Study '{study_name}' に有効な trial がありません。")
        return [], []

    orig_numbers = [t.number + 1 for t in trials]
    values = [float(t.value) for t in trials]

    filtered = [(n, v) for n, v in zip(orig_numbers, values) if v <= threshold]
    if not filtered:
        print(f"[WARN] Study '{study_name}' は全 trial が threshold({threshold}) 超えです。")
        return [], []

    orig_kept, vals_kept = zip(*filtered)

    if max_trials is not None:
        orig_kept = orig_kept[:max_trials]
        vals_kept = vals_kept[:max_trials]

    xs = list(range(1, len(vals_kept) + 1))
    ys = list(vals_kept)

    best_idx = int(np.argmin(ys))
    best_x = xs[best_idx]
    best_y = ys[best_idx]
    best_orig = orig_kept[best_idx]
    print(f"[INFO] Study '{study_name}': best DoA-A = {best_y:.3f}, "
          f"renumbered_trial={best_x}, orig_trial={best_orig}")

    return xs, ys


# ===================== メイン =====================

def main():
    parser = argparse.ArgumentParser(
        description="AVR と NAF の DoA-A(trial) を1枚にプロットするスクリプト"
    )

    # --- AVR 系 (複数指定可・logdirごとに trial 範囲/フォーマットを指定) ---
    parser.add_argument(
        "--avr_logdir", action="append", default=[],
        help="AVR系ログのルートディレクトリ (例: logs/real_exp)。複数指定可。"
    )
    parser.add_argument(
        "--avr_label", action="append", default=[],
        help="AVR系カーブの凡例ラベル (例: AVR)。--avr_logdir と同数だけ指定。"
    )
    parser.add_argument(
        "--avr_trial_begin", action="append", type=int, default=[],
        help="各AVR logdirに対応する trial 開始番号 (例: 1)。--avr_logdir と同数だけ指定。"
    )
    parser.add_argument(
        "--avr_trial_end", action="append", type=int, default=[],
        help="各AVR logdirに対応する trial 終了番号 (inclusive)。--avr_logdir と同数だけ指定。"
    )
    parser.add_argument(
        "--avr_trial_fmt", action="append", type=str, default=[],
        help='各AVR logdirに対応する trial名フォーマット (例: "Real_exp_param_{i}_1")。'
    )
    parser.add_argument(
        "--avr_algo_name", type=str, default="NormMUSIC",
        help="AVR側 pkl 内のアルゴリズム名 (例: NormMUSIC)"
    )
    parser.add_argument(
        "--avr_error_key", type=str, default="pred_vs_gt_error",
        help="AVR側の DoA-A 誤差キー (例: pred_vs_gt_error)"
    )
    parser.add_argument(
        "--avr_skip_missing", action="store_true",
        help="AVRの trial で DoA結果が無い場合、その trial をスキップ (デフォルトは NaN で残す)"
    )

    # --- NAF 系 (複数指定可) ---
    parser.add_argument(
        "--naf_storage", action="append", default=[],
        help="NAF系 Optuna storage URL (例: sqlite:///path/to/naf.db)。複数指定可。"
    )
    parser.add_argument(
        "--naf_study", action="append", default=[],
        help="NAF系 Optuna study 名 (例: real_exp_optuna)。--naf_storage と同数だけ指定。"
    )
    parser.add_argument(
        "--naf_label", action="append", default=[],
        help="NAF系カーブの凡例ラベル (例: NAF)。--naf_storage と同数だけ指定。"
    )
    parser.add_argument(
        "--naf_threshold", type=float, default=90.0,
        help="NAFの trial を残すための DoA-A しきい値 (value <= threshold を採用)"
    )
    parser.add_argument(
        "--naf_max_trials", type=int, default=300,
        help="NAF側で表示する trial の最大数 (先頭から). None なら制限なし"
    )

    # --- 共通 ---
    parser.add_argument(
        "--out_png", type=str, required=True,
        help="出力PNGパス"
    )
    parser.add_argument(
        "--ymin", type=float, default=0.0,
        help="y軸最小値 (DoA-A)"
    )
    parser.add_argument(
        "--ymax", type=float, default=120.0,
        help="y軸最大値 (DoA-A)"
    )
    args = parser.parse_args()

    # === 引数整合性チェック (AVR側) ===
    n_avr = len(args.avr_logdir)
    if not (len(args.avr_label) == len(args.avr_trial_begin) ==
            len(args.avr_trial_end) == len(args.avr_trial_fmt) == n_avr):
        raise ValueError(
            "AVR側の引数数が一致していません。\n"
            "必須: --avr_logdir, --avr_label, --avr_trial_begin, --avr_trial_end, --avr_trial_fmt が同数"
        )

    # === 引数整合性チェック (NAF側) ===
    n_naf = len(args.naf_storage)
    if not (len(args.naf_study) == len(args.naf_label) == n_naf):
        raise ValueError(
            "NAF側の引数数が一致していません。\n"
            "必須: --naf_storage, --naf_study, --naf_label が同数"
        )

    plt.figure(figsize=(10, 6))

    # ---------- AVR 側カーブ ----------
    for logdir, label, b, e, fmt in zip(
        args.avr_logdir,
        args.avr_label,
        args.avr_trial_begin,
        args.avr_trial_end,
        args.avr_trial_fmt,
    ):
        # logdirごとの trial_names を生成
        trial_names = [fmt.format(i=i) for i in range(b, e + 1)]

        xs, ys = collect_avr_doa_curve_for_logdir(
            logdir=logdir,
            trial_names=trial_names,
            algo_name=args.avr_algo_name,
            error_key=args.avr_error_key,
            skip_missing=args.avr_skip_missing,
        )
        if len(xs) == 0:
            print(f"[WARN] AVR logdir '{logdir}' から有効なデータが得られませんでした。スキップします。")
            continue

        plt.plot(xs, ys, label=label)  # 折れ線のみ（markerなし）

    # ---------- NAF 側カーブ ----------
    for storage, study, label in zip(args.naf_storage, args.naf_study, args.naf_label):
        xs, ys = collect_naf_doa_curve_from_optuna(
            storage=storage,
            study_name=study,
            threshold=args.naf_threshold,
            max_trials=args.naf_max_trials,
        )
        if len(xs) == 0:
            print(f"[WARN] NAF study '{study}' から有効なデータが得られませんでした。スキップします。")
            continue

        plt.plot(xs, ys, label=label)  # 折れ線のみ（markerなし）

    # 軸・凡例など
    plt.xlabel("Trial Index", fontsize=12)
    plt.ylabel("DoA-A Error (deg)", fontsize=12)
    plt.ylim(args.ymin, args.ymax)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Trial-wise DoA-A (AVR & NAF)")

    out_path = os.path.expanduser(args.out_png)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Figure saved to: {out_path}")


if __name__ == "__main__":
    main()

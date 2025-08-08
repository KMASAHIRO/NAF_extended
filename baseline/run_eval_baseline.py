import os
import argparse
import pickle
import numpy as np
import torch
from eval_func import metric_cal, run_doa_on_pkl

def evaluate_all(pkl_path, fs=16000):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    pred_dict = data["pred"]
    gt_dict = data["gt"]

    ch_metrics = {
        "angle_error": [],
        "amp_error": [],
        "env_error": [],
        "t60_error": [],
        "edt_error": [],
        "c50_error": [],
        "multi_stft_loss": []
    }

    pred_all = []
    gt_all = []

    mic_array_size = None  # ← 自動取得する

    for key in sorted(pred_dict.keys()):
        pred = pred_dict[key]  # (C, T)
        gt = gt_dict[key]

        if mic_array_size is None:
            mic_array_size = pred.shape[0]

        for ch in range(mic_array_size):
            try:
                metrics = metric_cal(
                    ori_ir=gt[ch],
                    pred_ir=pred[ch],
                    fs=fs
                )
                for k, v in zip(ch_metrics.keys(), metrics[:7]):
                    ch_metrics[k].append(float(v))
            except Exception as e:
                print(f"Metric failed at {key} ch {ch}: {e}")

        pred_all.extend(pred)
        gt_all.extend(gt)

    pred_all = np.stack(pred_all, axis=0)
    gt_all = np.stack(gt_all, axis=0)

    # === 保存先構築 ===
    base_dir = os.path.dirname(pkl_path)
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    base_name = os.path.basename(pkl_path)
    eval_pkl_path = os.path.join(eval_dir, base_name.replace(".pkl", "_eval.pkl"))
    doa_pkl_path = os.path.join(eval_dir, base_name.replace(".pkl", "_doa.pkl"))

    # === DoA評価 ===
    doa_results = run_doa_on_pkl(
        pkl_path=pkl_path,
        fs=fs,
        mic_array_size=mic_array_size,
        save_path=doa_pkl_path
    )

    doa_errors = {
        algo: {
            "pred_vs_gt_error": np.mean([e for e in result["pred_vs_gt_error"] if e is not None]),
            "pred_vs_true_error": np.mean([e for e in result["pred_vs_true_error"] if e is not None]),
            "gt_vs_true_error": np.mean([e for e in result["gt_vs_true_error"] if e is not None]),
        }
        for algo, result in doa_results.items()
    }

    # === 出力表示 ===
    print("\n=== IR Metrics (Mean) ===")
    for k, v in ch_metrics.items():
        print(f"{k}: {np.nanmean(v):.4f}")

    print("\n=== DoA Errors (Mean) ===")
    for algo, err_dict in doa_errors.items():
        print(f"[{algo}] pred_vs_gt: {err_dict['pred_vs_gt_error']:.2f}°, "
              f"pred_vs_true: {err_dict['pred_vs_true_error']:.2f}°, "
              f"gt_vs_true: {err_dict['gt_vs_true_error']:.2f}°")

    # === 評価結果保存 ===
    result_all = {
        "IR_metrics": ch_metrics,
        "DoA_errors": doa_errors
    }

    with open(eval_pkl_path, "wb") as f:
        pickle.dump(result_all, f)
    print(f"\n✅ All evaluation results saved to {eval_pkl_path}")
    print(f"✅ DoA details saved to {doa_pkl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="補間済みpklファイル（pred + gt）")
    parser.add_argument("--fs", type=int, default=16000, help="サンプリングレート（default: 16000）")
    args = parser.parse_args()

    evaluate_all(
        pkl_path=args.pkl,
        fs=args.fs
    )

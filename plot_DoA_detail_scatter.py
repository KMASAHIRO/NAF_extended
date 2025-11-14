import os
import glob
import pickle
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def angular_error_deg(a, b):
    """
    角度 [deg] 同士の誤差（0〜360 wrap）を絶対値で返す。
    """
    diff = np.abs(a - b)
    return np.minimum(diff, 360 - diff)


def compute_doa_from_spec(spec, mic_pos, fs=16000, n_fft=512):
    """
    spec : (M, F, T) の複素 STFT
    mic_pos : (2, M) のマイク位置
    """
    doa = pra.doa.algorithms["NormMUSIC"](mic_pos, fs=fs, nfft=n_fft)
    doa.locate_sources(spec)
    return np.argmax(doa.grid.values)


def plot_doa_comparison_nao(yaml_path: str):
    config = load_yaml(yaml_path)

    base_dir = os.path.join(config["save_loc"], config["exp_name"], "val_results")
    out_dir = os.path.join(config["save_loc"], config["exp_name"])
    os.makedirs(out_dir, exist_ok=True)

    doa_ch = config["dir_ch"]

    npz_paths = sorted(glob.glob(os.path.join(base_dir, "val_epoch_*.npz")))

    results = []
    for path in npz_paths:
        data = np.load(path)
        pred_spec_all = data["pred_sig_spec"]
        ori_spec_all = data["ori_sig_spec"]
        pos_rx = data["position_rx"]  # shape depends on doa_ch
        pos_tx = data["position_tx"]

        pred_deg_list, gt_deg_list, true_deg_list = [], [], []

        if doa_ch > 1:
            # --- マルチチャンネルIRをそのままDoAに使う場合 ---
            G = pos_rx.shape[0]
            for i in range(G):
                pred_group = pred_spec_all[i * doa_ch: (i + 1) * doa_ch]  # (doa_ch, F, T)
                ori_group = ori_spec_all[i * doa_ch: (i + 1) * doa_ch]
                mic_center = pos_rx[i]         # (2,) : すでにアレイ中心
                tx = pos_tx[i]                 # (2,)

                dx, dy = tx[0] - mic_center[0], tx[1] - mic_center[1]
                true_rad = np.arctan2(dy, dx)
                true_deg = np.degrees(true_rad) % 360

                mic_array = pra.beamforming.circular_2D_array(
                    center=mic_center,
                    M=doa_ch,
                    radius=0.0365,
                    phi0=np.pi / 2
                )

                pred_deg = compute_doa_from_spec(pred_group, mic_array)
                gt_deg = compute_doa_from_spec(ori_group, mic_array)

                pred_deg_list.append(pred_deg)
                gt_deg_list.append(gt_deg)
                true_deg_list.append(true_deg)

        else:
            # --- dir_ch == 1 のとき: 仮想円形アレイ8chを構成 ---
            M = 8
            N = pred_spec_all.shape[0]
            G = N // M

            for g in range(G):
                idxs = np.arange(g * M, (g + 1) * M)
                pred_group = pred_spec_all[idxs]  # (M, F, T)
                ori_group = ori_spec_all[idxs]
                rx_group = pos_rx[idxs]           # (M, 2)
                tx_group = pos_tx[idxs]           # (M, 2) だが全て同一想定

                mic_center = np.mean(rx_group, axis=0)
                tx = tx_group[0]  # どのTxでも同じはず

                dx, dy = tx[0] - mic_center[0], tx[1] - mic_center[1]
                true_rad = np.arctan2(dy, dx)
                true_deg = np.degrees(true_rad) % 360

                mic_array = pra.beamforming.circular_2D_array(
                    center=mic_center,
                    M=M,
                    radius=0.0365,
                    phi0=np.pi / 2
                )

                pred_deg = compute_doa_from_spec(pred_group, mic_array)
                gt_deg = compute_doa_from_spec(ori_group, mic_array)

                pred_deg_list.append(pred_deg)
                gt_deg_list.append(gt_deg)
                true_deg_list.append(true_deg)

        pred_deg_arr = np.array(pred_deg_list)
        gt_deg_arr = np.array(gt_deg_list)
        true_deg_arr = np.array(true_deg_list)

        # 平均誤差（Mean）
        err_pg = np.mean(angular_error_deg(pred_deg_arr, gt_deg_arr))
        err_pt = np.mean(angular_error_deg(pred_deg_arr, true_deg_arr))
        err_gt = np.mean(angular_error_deg(gt_deg_arr, true_deg_arr))

        results.append((path, err_pg, pred_deg_arr, gt_deg_arr, true_deg_arr, err_pt, err_gt))

    if not results:
        raise RuntimeError("No valid DoA data found.")

    # === pred_vs_gt_error が最小の「best」を採用 ===
    best_result = min(results, key=lambda x: x[1])
    path, _, pred_best, gt_best, true_best, _, _ = best_result
    epoch = int(os.path.basename(path).split("_")[-1].split(".")[0])

    # === 誤差配列と mean ± std を再計算 ===
    err_pg_arr = angular_error_deg(pred_best, gt_best)
    err_pt_arr = angular_error_deg(pred_best, true_best)
    err_gt_arr = angular_error_deg(gt_best, true_best)

    err_pg_mean = float(np.mean(err_pg_arr))
    err_pg_std = float(np.std(err_pg_arr))

    err_pt_mean = float(np.mean(err_pt_arr))
    err_pt_std = float(np.std(err_pt_arr))

    err_gt_mean = float(np.mean(err_gt_arr))
    err_gt_std = float(np.std(err_gt_arr))

    # === サブプロット用ヘルパ ===
    def draw_subplot_single(x, y, xlabel, ylabel, title, save_path):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(x, y, alpha=0.5)
        ax.plot([0, 360], [0, 360], 'r--')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved figure to {save_path}")

    # === 3種類の散布図をそれぞれ別PNGとして保存 ===
    save_path_pg = os.path.join(
        out_dir, f"doa_best_pred_vs_gt_epoch{epoch}.png"
    )
    save_path_pt = os.path.join(
        out_dir, f"doa_best_pred_vs_true_epoch{epoch}.png"
    )
    save_path_gt = os.path.join(
        out_dir, f"doa_best_gt_vs_true_epoch{epoch}.png"
    )

    title_pg = (
        f"Best (Epoch {epoch})\n"
        f"pred_vs_gt_error: {err_pg_mean:.2f}° ± {err_pg_std:.2f}°"
    )
    title_pt = (
        f"Best (Epoch {epoch})\n"
        f"pred_vs_true_error: {err_pt_mean:.2f}° ± {err_pt_std:.2f}°"
    )
    title_gt = (
        f"Best (Epoch {epoch})\n"
        f"gt_vs_true_error: {err_gt_mean:.2f}° ± {err_gt_std:.2f}°"
    )

    draw_subplot_single(gt_best, pred_best, "gt_deg", "pred_deg", title_pg, save_path_pg)
    draw_subplot_single(true_best, pred_best, "true_deg", "pred_deg", title_pt, save_path_pt)
    draw_subplot_single(true_best, gt_best, "true_deg", "gt_deg", title_gt, save_path_gt)

    # === mean_err ± std_err を txt に保存 ===
    stats_path = os.path.join(out_dir, f"doa_best_stats_epoch{epoch}.txt")
    with open(stats_path, "w") as f:
        f.write(f"Best epoch: {epoch}\n")
        f.write("\n")
        f.write(f"pred_vs_gt_error_mean_deg  = {err_pg_mean:.6f}\n")
        f.write(f"pred_vs_gt_error_std_deg   = {err_pg_std:.6f}\n")
        f.write(f"pred_vs_true_error_mean_deg = {err_pt_mean:.6f}\n")
        f.write(f"pred_vs_true_error_std_deg  = {err_pt_std:.6f}\n")
        f.write(f"gt_vs_true_error_mean_deg   = {err_gt_mean:.6f}\n")
        f.write(f"gt_vs_true_error_std_deg    = {err_gt_std:.6f}\n")
        f.write("\n")
        f.write("Summary (mean ± std):\n")
        f.write(f"  pred_vs_gt_error   = {err_pg_mean:.2f}° ± {err_pg_std:.2f}°\n")
        f.write(f"  pred_vs_true_error = {err_pt_mean:.2f}° ± {err_pt_std:.2f}°\n")
        f.write(f"  gt_vs_true_error   = {err_gt_mean:.2f}° ± {err_gt_std:.2f}°\n")

    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoA Evaluation for NAF")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    plot_doa_comparison_nao(args.config)

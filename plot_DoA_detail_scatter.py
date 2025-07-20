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
    return np.minimum(np.abs(a - b), 360 - np.abs(a - b))

def compute_doa_from_spec(spec, mic_pos, fs=16000, n_fft=512):
    doa = pra.doa.algorithms["NormMUSIC"](mic_pos, fs=fs, nfft=n_fft)
    doa.locate_sources(spec)
    return np.argmax(doa.grid.values)

def plot_doa_comparison_nao(yaml_path: str):
    config = load_yaml(yaml_path)

    base_dir = os.path.join(config["save_loc"], config["exp_name"], "val_results")
    save_path = os.path.join(config["save_loc"], config["exp_name"], "doa_detail_scatter.png")
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
            G = pred_spec_all.shape[0]
            for i in range(G):
                pred_group = pred_spec_all[i]  # (doa_ch, F, T)
                ori_group = ori_spec_all[i]
                mic_center = pos_rx[i]         # already center position
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
            M = 8
            N = pred_spec_all.shape[0]
            G = N // M

            for g in range(G):
                idxs = np.arange(g * M, (g + 1) * M)
                pred_group = pred_spec_all[idxs]
                ori_group = ori_spec_all[idxs]
                rx_group = pos_rx[idxs]
                tx_group = pos_tx[idxs]
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

        err_pg = np.mean(angular_error_deg(pred_deg_arr, gt_deg_arr))
        err_pt = np.mean(angular_error_deg(pred_deg_arr, true_deg_arr))
        err_gt = np.mean(angular_error_deg(gt_deg_arr, true_deg_arr))

        results.append((path, err_pg, pred_deg_arr, gt_deg_arr, true_deg_arr, err_pt, err_gt))

    if not results:
        raise RuntimeError("No valid DoA data found.")

    best_result = min(results, key=lambda x: x[1])
    last_result = results[-1]

    def draw_subplot(ax, x, y, xlabel, ylabel, title):
        ax.scatter(x, y, alpha=0.5)
        ax.plot([0, 360], [0, 360], 'r--')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)

    fig, axs = plt.subplots(2, 3, figsize=(21, 14))

    for i, (path, err_pg, pred, gt, true, err_pt, err_gt) in enumerate([best_result, last_result]):
        epoch = int(os.path.basename(path).split("_")[-1].split(".")[0])
        label = "Best" if i == 0 else "Last"

        draw_subplot(axs[i, 0], gt, pred, "gt_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_gt_error: {err_pg:.2f}°")
        draw_subplot(axs[i, 1], true, pred, "true_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_true_error: {err_pt:.2f}°")
        draw_subplot(axs[i, 2], true, gt, "true_deg", "gt_deg",
                     f"{label} (Epoch {epoch})\ngt_vs_true_error: {err_gt:.2f}°")

    fig.suptitle("DoA Results (NormMUSIC, NAF)", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoA Evaluation for NAF")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    plot_doa_comparison_nao(args.config)

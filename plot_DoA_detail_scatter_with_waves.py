import os
import glob
import pickle
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pyroomacoustics as pra
import librosa  # ★ 追加

# 全体のデフォルトは大きくしない or 控えめに
plt.rcParams["font.size"] = 20

# 軸ラベル・タイトルだけ大きくする
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["axes.titlesize"] = 25

# tick ラベルは固定サイズにしておく（ここを変えないのがポイント）
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def angular_error_deg(a, b):
    """
    角度 [deg] 同士の誤差（0〜360 wrap）を絶対値で返す。
    """
    diff = np.abs(a - b)
    return np.minimum(diff, 360 - diff)

def compute_doa_from_spec(spec, mic_pos, algo_name="NormMUSIC", fs=16000, n_fft=512):
    """
    spec : (M, F, T) の複素 STFT
    mic_pos : (2, M) のマイク位置
    algo_name : "NormMUSIC", "SRP", "FRIDA" など
    戻り値: (推定角インデックス, 全グリッド値)
    """
    doa = pra.doa.algorithms[algo_name](mic_pos, fs=fs, nfft=n_fft)
    doa.locate_sources(spec)

    if algo_name == "FRIDA":
        # FRIDA は grid.values ではなく dirty image を使う
        dirty = doa._gen_dirty_img()
        values = dirty
        est_idx = int(np.argmax(np.abs(dirty)))
    else:
        values = doa.grid.values
        est_idx = int(np.argmax(values))

    return est_idx, values

def percentile_trim_mean_std(err_arr: np.ndarray, percentile: float):
    """
    誤差配列 err_arr に対して、指定 percentile（例: 95, 90）を閾値とし、
    その値を超える誤差を除外して mean/std を返す。

      - 95 → 95パーセンタイル以下のみを使用（≒上位5%を除く）
      - 90 → 90パーセンタイル以下のみを使用（≒上位10%を除く）
    """
    if err_arr.size == 0:
        return np.nan, np.nan
    thr = np.percentile(err_arr, percentile)
    mask = err_arr <= thr
    if not np.any(mask):
        # さすがに全部除外は避けて、最小値だけ残す
        idx_min = int(np.argmin(err_arr))
        trimmed = err_arr[idx_min:idx_min + 1]
    else:
        trimmed = err_arr[mask]
    return float(np.mean(trimmed)), float(np.std(trimmed))


def istft_ir(spec: np.ndarray, nfft=512, hop=128, win="hann") -> np.ndarray:
    """
    spec: (G, F, T) complex
    return: (G, Nt)
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


def plot_8ch_waveforms(gt_wave: np.ndarray,
                       pred_wave: np.ndarray,
                       out_path: str,
                       title: str = ""):
    """
    8ch波形を 4列×2行 にプロットする。
    gt_wave, pred_wave : (C, Nt)
    """
    n_ch, Nt = gt_wave.shape
    n_plot = min(8, n_ch)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()

    t = np.arange(Nt)  # サンプルインデックス（時間軸にしてもよい）

    for ch in range(n_plot):
        ax = axes[ch]
        ax.plot(t, gt_wave[ch], color="orange", label="正解波形")
        ax.plot(t, pred_wave[ch], color="blue", label="予測波形")
        ax.set_title(f"Ch {ch}")
        if ch // 4 == 1:
            ax.set_xlabel("サンプルインデックス")
        if ch % 4 == 0:
            ax.set_ylabel("振幅")
        ax.grid(True, alpha=0.3)

    # 凡例は1つだけ
    axes[0].legend(loc="upper right")

    if title:
        fig.suptitle(title)

    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Saved waveform plot to {out_path}")


def plot_doa_comparison_nao(yaml_path: str):
    config = load_yaml(yaml_path)

    base_dir = os.path.join(config["save_loc"], config["exp_name"], "val_results")
    out_dir = os.path.join(config["save_loc"], config["exp_name"])
    os.makedirs(out_dir, exist_ok=True)

    doa_ch = config["dir_ch"]

    npz_paths = sorted(glob.glob(os.path.join(base_dir, "val_epoch_*.npz")))
    if not npz_paths:
        raise RuntimeError(f"No val_epoch_*.npz found under: {base_dir}")

    stats_path = None  # NormMUSIC で確定させて、SRP/FRIDA は追記
    norm_epoch_for_stats = None

    # アルゴリズムごとに同じ処理を回す
    # for algo_name in ["NormMUSIC", "SRP", "FRIDA"]:
    for algo_name in ["NormMUSIC"]:
        print(f"=== Running DoA evaluation with {algo_name} ===")

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

                    pred_deg, _ = compute_doa_from_spec(pred_group, mic_array, algo_name=algo_name)
                    gt_deg, _ = compute_doa_from_spec(ori_group, mic_array, algo_name=algo_name)

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

                    pred_deg, _ = compute_doa_from_spec(pred_group, mic_array, algo_name=algo_name)
                    gt_deg, _ = compute_doa_from_spec(ori_group, mic_array, algo_name=algo_name)

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
            raise RuntimeError(f"No valid DoA data found for {algo_name}.")

        # === pred_vs_gt_error が最小の「best」を採用 ===
        best_result = min(results, key=lambda x: x[1])
        path, _, pred_best, gt_best, true_best, _, _ = best_result
        epoch = int(os.path.basename(path).split("_")[-1].split(".")[0])

        best_data = np.load(path)
        pred_spec_best_all = best_data["pred_sig_spec"]
        ori_spec_best_all  = best_data["ori_sig_spec"]

        # NormMUSIC の epoch を stats ファイル名に使う
        if algo_name == "NormMUSIC":
            norm_epoch_for_stats = epoch
            stats_path = os.path.join(out_dir, f"doa_best_stats_epoch{epoch}.txt")

        # === 誤差配列と mean ± std を再計算 ===
        err_pg_arr = angular_error_deg(pred_best, gt_best)    # DoA-A
        err_pt_arr = angular_error_deg(pred_best, true_best)  # DoA-B
        err_gt_arr = angular_error_deg(gt_best, true_best)

        # 全サンプルでの平均・標準偏差
        err_pg_mean = float(np.mean(err_pg_arr))
        err_pg_std = float(np.std(err_pg_arr))

        err_pt_mean = float(np.mean(err_pt_arr))
        err_pt_std = float(np.std(err_pt_arr))

        err_gt_mean = float(np.mean(err_gt_arr))
        err_gt_std = float(np.std(err_gt_arr))

        # === 90% / 95% タイルでトリムした mean/std を計算 ===
        err_pg_mean_95, err_pg_std_95 = percentile_trim_mean_std(err_pg_arr, 95.0)
        err_pt_mean_95, err_pt_std_95 = percentile_trim_mean_std(err_pt_arr, 95.0)
        err_gt_mean_95, err_gt_std_95 = percentile_trim_mean_std(err_gt_arr, 95.0)

        err_pg_mean_90, err_pg_std_90 = percentile_trim_mean_std(err_pg_arr, 90.0)
        err_pt_mean_90, err_pt_std_90 = percentile_trim_mean_std(err_pt_arr, 90.0)
        err_gt_mean_90, err_gt_std_90 = percentile_trim_mean_std(err_gt_arr, 90.0)

        # === パーセンタイル（25, 50, 75）を計算 ===
        pg_p25, pg_p50, pg_p75 = np.percentile(err_pg_arr, [25, 50, 75])
        pt_p25, pt_p50, pt_p75 = np.percentile(err_pt_arr, [25, 50, 75])
        gt_p25, gt_p50, gt_p75 = np.percentile(err_gt_arr, [25, 50, 75])

        # === サンプルごとの true/gt/pred と誤差を CSV に保存 ===
        if algo_name == "NormMUSIC":
            algo_suffix = ""
        else:
            algo_suffix = f"_{algo_name.lower()}"

        # NormMUSIC だけ suffix なしで元のファイル名
        if algo_name == "NormMUSIC":
            sample_csv_path = os.path.join(out_dir, f"doa_best_samples_epoch{epoch}.csv")
        else:
            sample_csv_path = os.path.join(out_dir, f"doa_best_samples{algo_suffix}_epoch{epoch}.csv")

        rows = np.column_stack([
            true_best,
            gt_best,
            pred_best,
            err_pg_arr,
            err_pt_arr,
            err_gt_arr,
        ])
        header = ",".join([
            "true_deg",
            "gt_deg",
            "pred_deg",
            "err_pred_vs_gt_deg",    # DoA-A
            "err_pred_vs_true_deg",  # DoA-B
            "err_gt_vs_true_deg",
        ])
        np.savetxt(sample_csv_path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
        print(f"[{algo_name}] Saved per-sample CSV to {sample_csv_path}")

        # === 2つの散布図 (DoA-A / DoA-B) を1枚にまとめて保存 ===
        if algo_name == "NormMUSIC":
            fig_path = os.path.join(out_dir, f"doa_best_doaA_doaB_epoch{epoch}.png")
        else:
            fig_path = os.path.join(out_dir, f"doa_best_doaA_doaB_{algo_name.lower()}_epoch{epoch}.png")

        # ★ ここで constrained_layout=True を指定
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

        # 左: DoA-A (pred vs gt)
        ax = axes[0]
        ax.scatter(gt_best, pred_best, alpha=0.5)
        ax.plot([0, 360], [0, 360], 'r--')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("正解定位方向 [°]")
        ax.set_ylabel("予測定位方向 [°]")
        ax.set_xticks(np.arange(0, 361, 50))
        ax.set_yticks(np.arange(0, 361, 50))
        ax.grid(True, alpha=0.3)

        # 右: DoA-B (pred vs true)
        ax = axes[1]
        ax.scatter(true_best, pred_best, alpha=0.5)
        ax.plot([0, 360], [0, 360], 'r--')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("真の音源方向 [°]")
        ax.set_ylabel("予測定位方向 [°]")
        ax.set_xticks(np.arange(0, 361, 50))
        ax.set_yticks(np.arange(0, 361, 50))
        ax.grid(True, alpha=0.3)

        # ★ tight_layout() は呼ばず、bbox_inches="tight" で保存
        plt.savefig(fig_path, bbox_inches="tight", dpi=100)
        plt.close()
        print(f"[{algo_name}] Saved DoA-A/DoA-B figure to {fig_path}")

        # === 8ch 波形プロット + 方向情報 ===
        save_dir_all = os.path.join(out_dir, "waveforms_all_samples")
        os.makedirs(save_dir_all, exist_ok=True)

        # DoA-A の昇順ソート
        sorted_indices = np.argsort(err_pg_arr)  # rank → sample index

        # 方向情報 txt
        dir_all_txt = os.path.join(save_dir_all, "directions_all_samples.txt")
        with open(dir_all_txt, "w", encoding="utf-8") as ftxt:
            ftxt.write("===== 全サンプル方向情報 (DoA-A 昇順) =====\n\n")

        # 全サンプル処理ループ
        for rank, i in enumerate(sorted_indices):
            # i 番目の観測サンプルについて pred/gt のスペクトログラムを取得
            if doa_ch > 1:
                pred_group = pred_spec_best_all[i * doa_ch : (i + 1) * doa_ch]
                ori_group  = ori_spec_best_all[i * doa_ch : (i + 1) * doa_ch]
            else:
                M = 8
                idxs = np.arange(i * M, (i + 1) * M)
                pred_group = pred_spec_best_all[idxs]
                ori_group  = ori_spec_best_all[idxs]

            # ISTFT 波形
            gt_wave  = istft_ir(ori_group)
            pred_wave = istft_ir(pred_group)

            # 波形画像の保存
            out_png = os.path.join(
                save_dir_all, 
                f"doaA_rank_{rank:04d}_idx_{i}.png"
            )
            plot_8ch_waveforms(
                gt_wave,
                pred_wave,
                out_png,
                title=f"DoA-A rank={rank}  idx={i}"
            )

            # 方向情報書き込み
            true_i = true_best[i]
            gt_i   = gt_best[i]
            pred_i = pred_best[i]

            with open(dir_all_txt, "a", encoding="utf-8") as ftxt:
                ftxt.write(f"=== sample rank {rank} (idx={i}) ===\n")
                ftxt.write(f"true_direction_deg (真の音源方向)     = {true_i:.6f}\n")
                ftxt.write(f"gt_direction_deg   (正解波形のDoA)     = {gt_i:.6f}\n")
                ftxt.write(f"pred_direction_deg (予測波形のDoA)     = {pred_i:.6f}\n")
                ftxt.write("\n")

        print(f"[{algo_name}] 全サンプルの波形を {save_dir_all} に保存しました")
        print(f"[{algo_name}] 方向情報を {dir_all_txt} に保存しました")

        # === mean_err ± std_err とパーセンタイルを txt に保存 ===
        if stats_path is None:
            raise RuntimeError("stats_path is not defined. NormMUSIC should run first.")

        # NormMUSIC は新規作成、それ以外は追記
        mode = "w" if algo_name == "NormMUSIC" else "a"
        with open(stats_path, mode) as f:
            f.write(f"Algorithm: {algo_name}\n")
            f.write(f"Best epoch: {epoch}\n")
            f.write("\n")
            f.write("=== 全サンプル ===\n")
            f.write(f"pred_vs_gt_error_mean_deg       = {err_pg_mean:.6f}\n")
            f.write(f"pred_vs_gt_error_std_deg        = {err_pg_std:.6f}\n")
            f.write(f"pred_vs_true_error_mean_deg     = {err_pt_mean:.6f}\n")
            f.write(f"pred_vs_true_error_std_deg      = {err_pt_std:.6f}\n")
            f.write(f"gt_vs_true_error_mean_deg       = {err_gt_mean:.6f}\n")
            f.write(f"gt_vs_true_error_std_deg        = {err_gt_std:.6f}\n")
            f.write("\n")
            f.write("=== 95th percentile 以下のみ（上位5%除外に相当） ===\n")
            f.write(f"pred_vs_gt_error_mean_deg_95pctl   = {err_pg_mean_95:.6f}\n")
            f.write(f"pred_vs_gt_error_std_deg_95pctl    = {err_pg_std_95:.6f}\n")
            f.write(f"pred_vs_true_error_mean_deg_95pctl = {err_pt_mean_95:.6f}\n")
            f.write(f"pred_vs_true_error_std_deg_95pctl  = {err_pt_std_95:.6f}\n")
            f.write(f"gt_vs_true_error_mean_deg_95pctl   = {err_gt_mean_95:.6f}\n")
            f.write(f"gt_vs_true_error_std_deg_95pctl    = {err_gt_std_95:.6f}\n")
            f.write("\n")
            f.write("=== 90th percentile 以下のみ（上位10%除外に相当） ===\n")
            f.write(f"pred_vs_gt_error_mean_deg_90pctl   = {err_pg_mean_90:.6f}\n")
            f.write(f"pred_vs_gt_error_std_deg_90pctl    = {err_pg_std_90:.6f}\n")
            f.write(f"pred_vs_true_error_mean_deg_90pctl = {err_pt_mean_90:.6f}\n")
            f.write(f"pred_vs_true_error_std_deg_90pctl  = {err_pt_std_90:.6f}\n")
            f.write(f"gt_vs_true_error_mean_deg_90pctl   = {err_gt_mean_90:.6f}\n")
            f.write(f"gt_vs_true_error_std_deg_90pctl    = {err_gt_std_90:.6f}\n")
            f.write("\n")
            f.write("=== パーセンタイル（25, 50, 75） ===\n")
            f.write(f"pred_vs_gt_error_p25_deg = {pg_p25:.6f}\n")
            f.write(f"pred_vs_gt_error_p50_deg = {pg_p50:.6f}  # median\n")
            f.write(f"pred_vs_gt_error_p75_deg = {pg_p75:.6f}\n")
            f.write(f"pred_vs_true_error_p25_deg = {pt_p25:.6f}\n")
            f.write(f"pred_vs_true_error_p50_deg = {pt_p50:.6f}  # median\n")
            f.write(f"pred_vs_true_error_p75_deg = {pt_p75:.6f}\n")
            f.write(f"gt_vs_true_error_p25_deg = {gt_p25:.6f}\n")
            f.write(f"gt_vs_true_error_p50_deg = {gt_p50:.6f}  # median\n")
            f.write(f"gt_vs_true_error_p75_deg = {gt_p75:.6f}\n")
            f.write("\n")
            f.write("Summary (mean ± std):\n")
            f.write(f"  pred_vs_gt_error   = {err_pg_mean:.2f}° ± {err_pg_std:.2f}°\n")
            f.write(f"  pred_vs_true_error = {err_pt_mean:.2f}° ± {err_pt_std:.2f}°\n")
            f.write(f"  gt_vs_true_error   = {err_gt_mean:.2f}° ± {err_gt_std:.2f}°\n")
            f.write("\n")
            f.write("  pred_vs_gt_error (≤95th)   = {0:.2f}° ± {1:.2f}°\n".format(err_pg_mean_95, err_pg_std_95))
            f.write("  pred_vs_true_error (≤95th) = {0:.2f}° ± {1:.2f}°\n".format(err_pt_mean_95, err_pt_std_95))
            f.write("  gt_vs_true_error (≤95th)   = {0:.2f}° ± {1:.2f}°\n".format(err_gt_mean_95, err_gt_std_95))
            f.write("  pred_vs_gt_error (≤90th)   = {0:.2f}° ± {1:.2f}°\n".format(err_pg_mean_90, err_pg_std_90))
            f.write("  pred_vs_true_error (≤90th) = {0:.2f}° ± {1:.2f}°\n".format(err_pt_mean_90, err_pt_std_90))
            f.write("  gt_vs_true_error (≤90th)   = {0:.2f}° ± {1:.2f}°\n".format(err_gt_mean_90, err_gt_std_90))
            f.write("\n")
            f.write("=== 誤差一覧（昇順） ===\n")
            f.write("pred_vs_gt_error_sorted_deg = [")
            f.write(", ".join(f"{v:.6f}" for v in np.sort(err_pg_arr)))
            f.write("]\n")

            f.write("pred_vs_true_error_sorted_deg = [")
            f.write(", ".join(f"{v:.6f}" for v in np.sort(err_pt_arr)))
            f.write("]\n")

            f.write("gt_vs_true_error_sorted_deg = [")
            f.write(", ".join(f"{v:.6f}" for v in np.sort(err_gt_arr)))
            f.write("]\n")
            f.write("\n\n")

        print(f"[{algo_name}] Appended stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoA Evaluation for NAF (NormMUSIC / SRP / FRIDA)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    plot_doa_comparison_nao(args.config)

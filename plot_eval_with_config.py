import os
import re
import glob
import yaml
import pickle
import argparse
import matplotlib.pyplot as plt

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def plot_naf_loss_and_doa(config_path):
    # === YAML読込み ===
    config = load_yaml(config_path)
    save_loc = config["save_loc"]
    exp_name = config["exp_name"]

    loss_dir = os.path.join(save_loc, exp_name, "loss_values")

    # === loss_epoch_*.pkl を取得・並べ替え ===
    pkl_files = sorted(
        glob.glob(os.path.join(loss_dir, "loss_epoch_*.pkl")),
        key=lambda f: int(re.findall(r"loss_epoch_(\d+).pkl", f)[0])
    )

    epochs = []
    train_loss = []
    val_loss = []
    doa_errors = []

    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        epochs.append(data["epoch"])
        train_loss.append(data["loss"])
        val_loss.append(data["loss_val"])
        doa_errors.append(data["DoA_err"])  # pred_vs_gt_error 相当

    # === 描画 ===
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左軸（Loss）
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="black")
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue")
    ax1.plot(epochs, val_loss, label="Val Loss", color="orange")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True)

    # 右軸（DoA Error）
    ax2 = ax1.twinx()
    ax2.set_ylabel("DoA Error (°)")
    ax2.plot(epochs, doa_errors, label="DoA Error", color="green")
    ax2.set_ylim(0, 120)
    ax2.tick_params(axis='y')

    # 凡例統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Loss and DoA Error (pred_vs_gt_error) over Epochs")
    plt.tight_layout()

    # 保存
    output_path = os.path.join(save_loc, exp_name, "loss_and_doa_plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")

# === CLIエントリポイント ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NAF Loss and DoA Error over Epochs")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()
    plot_naf_loss_and_doa(args.config)

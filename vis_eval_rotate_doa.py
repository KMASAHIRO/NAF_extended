# plot_rotate_pred_vs_true.py
import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

def angular_error_deg(a, b):
    a = np.asarray(a) % 360.0
    b = np.asarray(b) % 360.0
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="eval_rotate_doa.py の出力ディレクトリ")
    ap.add_argument("--save_name", type=str, default="pred_vs_true.png")
    args = ap.parse_args()

    npz_paths = sorted(glob.glob(os.path.join(args.out_dir, "*.npz")))
    if not npz_paths:
        raise RuntimeError("No npz files found in out_dir.")

    all_pred, all_true = [], []
    for p in npz_paths:
        data = np.load(p)
        pred = data["pred_deg"].astype(int)
        true = data["true_deg"].astype(int)
        all_pred.append(pred); all_true.append(true)

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    mae = float(angular_error_deg(pred, true).mean())

    plt.figure(figsize=(7,7))
    plt.scatter(true, pred, alpha=0.5)
    plt.plot([0,360], [0,360], 'r--', linewidth=1)
    plt.xlim(0,360); plt.ylim(0,360)
    plt.xticks(np.arange(0, 361, 50))
    plt.yticks(np.arange(0, 361, 50))
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel("真の音源方向 [°]")
    plt.ylabel("予測定位方向 [°]")
    #plt.title(f"pred vs true (N={len(pred)}, MAE={mae:.2f}°)")
    save_path = os.path.join(args.out_dir, args.save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()

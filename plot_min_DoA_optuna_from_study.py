import optuna
import matplotlib.pyplot as plt
import argparse
import os

def plot_doa_vs_trial(storage: str, study_name: str, out_dir: str, threshold: float = 90, max_trials: int = None):
    """
    Optunaのstudy結果を読み込み、
    trial番号とDoA_errの関係をプロットする。
    thresholdを超える値を除外し、再番号付け後にmax_trialsまで表示。
    最良trialとその元のtrial indexを表示。
    """

    # === Study読み込み ===
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [t for t in study.trials if t.value is not None]
    if not trials:
        print(f"⚠️ Study '{study_name}' に有効なtrial結果がありません。")
        return

    # === 全試行抽出 ===
    trial_numbers_all = [t.number + 1 for t in trials]
    doa_errors_all = [t.value for t in trials]

    # === threshold適用 ===
    filtered_data = [(n, v) for n, v in zip(trial_numbers_all, doa_errors_all) if v <= threshold]
    if not filtered_data:
        print(f"⚠️ すべてのtrial値がthreshold({threshold})を超えています。")
        return

    filtered_numbers_orig, filtered_values = zip(*filtered_data)
    renumbered = list(range(1, len(filtered_values) + 1))

    # === max_trials制限 ===
    if max_trials is not None:
        renumbered = renumbered[:max_trials]
        filtered_values = filtered_values[:max_trials]
        filtered_numbers_orig = filtered_numbers_orig[:max_trials]

    # === ベストtrial ===
    best_idx = min(range(len(filtered_values)), key=lambda i: filtered_values[i])
    best_trial = renumbered[best_idx]                # 前倒し後の番号
    best_value = filtered_values[best_idx]
    best_orig_trial = filtered_numbers_orig[best_idx]  # 元のtrial index

    # === ベストtrial（上位5件まで表示） ===
    # 小さい順にソート（indexリスト）
    sorted_idx = sorted(range(len(filtered_values)), key=lambda i: filtered_values[i])

    # 上位5件（trial が5未満ならその分だけ）
    top_k = min(5, len(sorted_idx))
    top_sorted_idx = sorted_idx[:top_k]

    print("Top 5 trials (after renumbering):")
    for rank, idx in enumerate(top_sorted_idx, start=1):
        trial_num = renumbered[idx]              # 前倒し後番号
        val = filtered_values[idx]               # DoA error
        orig = filtered_numbers_orig[idx]        # 元のtrial番号
        print(f"  #{rank}: val={val:.3f}, orig_trial={orig}")    

    # === プロット ===
    plt.figure(figsize=(8, 5))
    plt.plot(renumbered, filtered_values, marker="o", label="DoA_err")
    plt.scatter(best_trial, best_value, color="red", s=80,
                label=f"Best = {best_trial} ({best_value:.3f})")
    plt.xlabel("Trial", fontsize=12)
    plt.ylabel("DoA_err", fontsize=12)
    plt.ylim(0, 100)
    plt.title(f"Optuna Study: {study_name}", fontsize=14)
    plt.grid(True)
    plt.legend()

    # === 保存 ===
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{study_name}_min_doa_plot_{max_trials}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Plot saved to: {save_path}")
    print(f"（除外条件: DoA_err > {threshold}, 最大表示trial: {max_trials or 'すべて'}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=90)
    parser.add_argument("--max_trials", type=int, default=300)
    args = parser.parse_args()

    args.out_dir = os.path.expanduser(args.out_dir)

    plot_doa_vs_trial(args.storage, args.study_name, args.out_dir,
                      threshold=args.threshold, max_trials=args.max_trials)

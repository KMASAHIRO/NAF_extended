import argparse
import subprocess
import yaml
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # YAML 読み込み
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # config ファイルを save_loc/exp_name 以下にコピー（← subprocess前に実行）
    save_root = os.path.join(config["save_loc"], config["exp_name"])
    os.makedirs(save_root, exist_ok=True)
    config_save_path = os.path.join(save_root, "config.yml")
    shutil.copy(args.config, config_save_path)
    print(f"Config file saved to: {config_save_path}")

    # subprocess 実行コマンドを構成
    cmd = [
        "python", "NAF_extended/model_pipeline/train/train.py",
        "--exp_name", config["exp_name"],
        "--coor_base", config["coor_base"],
        "--spec_base", config["spec_base"],
        "--phase_base", config["phase_base"],
        "--mean_std_base", config["mean_std_base"],
        "--phase_std_base", config["phase_std_base"],
        "--minmax_base", config["minmax_base"],
        "--wav_base", config["wav_base"],
        "--split_loc", config["split_loc"],
        "--gpus", str(config["gpus"]),
        "--dir_ch", str(config["dir_ch"]),
        "--max_len", str(config["max_len"]),
        "--layers", str(config["layers"]),
        "--layers_residual", str(config["layers_residual"]),
        "--features", str(config["features"]),
        "--epochs", str(config["epochs"]),
        "--reg_eps", str(config["reg_eps"]),
        "--lr_init", str(config["lr_init"]),
        "--lr_decay", str(config["lr_decay"]),
        "--phase_alpha", str(config["phase_alpha"]),
        "--mag_alpha", str(config["mag_alpha"]),
        "--batch_norm", config["batch_norm"],
        "--activation_func_name", config["activation_func_name"],
        "--save_loc", config["save_loc"]
    ]

    # 実行
    # 環境変数を準備
    env = os.environ.copy()
    naf_root = os.path.abspath("NAF_extended")
    env["PYTHONPATH"] = naf_root  # ← model_pipeline が import できるように

    # 実行（カレントディレクトリそのまま）
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()

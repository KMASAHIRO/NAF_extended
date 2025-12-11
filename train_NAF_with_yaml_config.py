#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
naf_train_once.py

NAF の設定 YAML を 1 つ受け取り，
その内容どおりに train.py を 1 回だけ実行するスクリプト。

- config YAML には，元の Optuna スクリプトと同じキー
  (exp_name, save_loc, coor_base, spec_base, phase_base, mean_std_base, phase_std_base,
   minmax_base, wav_base, split_loc, gpus, dir_ch, max_len, layers, layers_residual,
   features, epochs, reg_eps, lr_init, lr_decay, phase_alpha, mag_alpha,
   batch_norm, activation_func_name, pixel_count)
  が入っている前提。

- train.py は CLI 引数のみで設定を受け取る想定で，
  元の Optuna 用コードと同じ引数を渡す。
"""

import os
import sys
import yaml
import argparse
import subprocess
from typing import Dict, Any


def run_training_once(
    cfg: Dict[str, Any],
    naf_root: str,
    train_script_path: str,
) -> int:
    """
    1 回分の NAF 学習を subprocess.run() で train.py を直接叩いて実行する。

    Parameters
    ----------
    cfg : dict
        NAF の設定（YAML をロードした dict）。
    naf_root : str
        NAF_extended のルートディレクトリ。
        model_pipeline などを import できるように PYTHONPATH に通す。
    train_script_path : str
        NAF_extended/model_pipeline/train/train.py のパス。

    Returns
    -------
    int
        subprocess の returncode（0 が成功）。
    """

    env = os.environ.copy()
    env["PYTHONPATH"] = naf_root

    cmd = [
        "python",
        train_script_path,

        "--exp_name", cfg["exp_name"],
        "--coor_base", cfg["coor_base"],
        "--spec_base", cfg["spec_base"],
        "--phase_base", cfg["phase_base"],
        "--mean_std_base", cfg["mean_std_base"],
        "--phase_std_base", cfg["phase_std_base"],
        "--minmax_base", cfg["minmax_base"],
        "--wav_base", cfg["wav_base"],
        "--split_loc", cfg["split_loc"],

        "--gpus", str(cfg["gpus"]),
        "--dir_ch", str(cfg["dir_ch"]),
        "--max_len", str(cfg["max_len"]),
        "--layers", str(cfg["layers"]),
        "--layers_residual", str(cfg["layers_residual"]),
        "--features", str(cfg["features"]),
        "--epochs", str(cfg["epochs"]),
        "--reg_eps", str(cfg["reg_eps"]),
        "--lr_init", str(cfg["lr_init"]),
        "--lr_decay", str(cfg["lr_decay"]),
        "--phase_alpha", str(cfg["phase_alpha"]),
        "--mag_alpha", str(cfg["mag_alpha"]),
        "--batch_norm", str(cfg["batch_norm"]),
        "--activation_func_name", cfg["activation_func_name"],
        "--save_loc", cfg["save_loc"],
        "--pixel_count", str(cfg["pixel_count"]),
    ]

    print("[NAF] Running training subprocess:")
    print("  " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def save_config_copy(cfg: Dict[str, Any], filename: str = "naf_conf.yml") -> str:
    """
    cfg のコピーを <save_loc>/<exp_name>/filename に保存する。

    戻り値: 保存した YAML のパス。
    """
    exp_dir = os.path.join(cfg["save_loc"], cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    yaml_path = os.path.join(exp_dir, filename)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"[NAF] Saved config to {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()

    # ベースとなる NAF 設定 YAML
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="NAF 設定 YAML のパス",
    )

    # train.py の場所（直接叩く）
    parser.add_argument(
        "--train_script",
        type=str,
        required=True,
        help="NAF の train.py へのパス (例: ~/NAF_extended/model_pipeline/train/train.py)",
    )

    # PYTHONPATH に追加するルート (model_pipeline が import できるトップ)
    parser.add_argument(
        "--naf_root",
        type=str,
        default="NAF_extended",
        help="PYTHONPATH に追加するディレクトリ (例: ~/NAF_extended)",
    )

    # 必要に応じて exp_name を上書きしたい場合
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="config 内の exp_name を上書きしたい場合に指定",
    )

    args = parser.parse_args()

    # "~" 展開
    config_path = os.path.expanduser(args.config)
    train_script_path = os.path.expanduser(args.train_script)
    naf_root = os.path.expanduser(args.naf_root)

    # config のロード
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # exp_name を上書きしたい場合
    if args.exp_name is not None:
        print(f"[NAF] Override exp_name: {cfg.get('exp_name')} -> {args.exp_name}")
        cfg["exp_name"] = args.exp_name

    # config のコピーを保存（任意だが便利なので）
    save_config_copy(cfg, filename="naf_conf.yml")

    # 1 回だけ学習を実行
    ret = run_training_once(
        cfg=cfg,
        naf_root=naf_root,
        train_script_path=train_script_path,
    )

    if ret != 0:
        print(f"[NAF] WARNING: training subprocess exited with code {ret}")
    else:
        print("[NAF] Training finished successfully.")

    # スクリプト自体の終了コードとして返す
    sys.exit(ret)


if __name__ == "__main__":
    main()

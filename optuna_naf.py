import os
import re
import glob
import yaml
import pickle
import argparse
import subprocess
import optuna
from typing import Dict, Any


def make_trial_config(
    base_config: Dict[str, Any],
    base_start_index: int,
    trial_index: int,
    trial: optuna.trial.Trial,
) -> Dict[str, Any]:
    """
    base_configをコピーし、このtrial用のハイパラ・exp_nameに書き換えたconfigを返す。

    チューニング対象（ユーザ指定済み）:
      - layers:            4 ~ 12                (int)
      - layers_residual:   0 ~ 3                 (int)
      - features:          2^exp, exp=6..10 → 64~1024
      - reg_eps:           0.0 ~ 0.2             (float)
      - lr_init:           1e-5 ~ 1e-2           (log scale)
      - lr_decay:          0.05 ~ 0.5            (log scale)
      - phase_alpha:       0.1 ~ 10              (float)
      - pixel_count:       200 ~ 4000            (int)

    それ以外（save_loc, exp_name, coor_base, ... mag_alpha, batch_norm,
    activation_func_name, gpus, dir_ch, max_len, epochs など）は変更しない。
    """

    # deepcopy相当: dump->load
    cfg = yaml.load(yaml.dump(base_config), Loader=yaml.FullLoader)

    # ---- Optunaサジェスト ----
    # ネットワーク深さ
    layers = trial.suggest_int("layers", 4, 12)

    # 残差ブロック数
    layers_residual = trial.suggest_int("layers_residual", 0, 3)

    # features: 2^exp, exp=6..10 → {64,128,256,512,1024}
    features_exp = trial.suggest_int("features_exp", 6, 10)
    features = 2 ** features_exp

    # 位置ノイズなどの正則化
    reg_eps = trial.suggest_float("reg_eps", 0.0, 0.2)

    # 学習率など
    lr_init = trial.suggest_float("lr_init", 1e-5, 1e-2, log=True)
    lr_decay = trial.suggest_float("lr_decay", 0.05, 0.5, log=True)

    # 損失の重み
    phase_alpha = trial.suggest_float("phase_alpha", 0.1, 10.0)

    # 1 forward で扱う時間サンプル数(tiles)
    pixel_count = trial.suggest_int("pixel_count", 200, 4000)

    # ---- cfgへ反映 ----
    cfg["layers"] = layers
    cfg["layers_residual"] = layers_residual
    cfg["features"] = features
    cfg["reg_eps"] = reg_eps
    cfg["lr_init"] = lr_init
    cfg["lr_decay"] = lr_decay
    cfg["phase_alpha"] = phase_alpha
    cfg["pixel_count"] = pixel_count

    # ※ 他のキー（save_loc, exp_name, coor_base, spec_base, ...,
    #    mag_alpha, batch_norm, activation_func_name, epochs, etc.）は
    #    base_configの値をそのまま使う

    # ---- exp_nameユニーク化 (AVRスタイル: param_{trial_num}_1) ----
    trial_num = base_start_index + trial_index
    base_exp = cfg["exp_name"]

    # 既存の param_\d+_1 を置換、なければ末尾に追加
    new_exp = re.sub(r'param_\d+_1', f'param_{trial_num}_1', base_exp)
    if new_exp == base_exp:
        new_exp = f"{base_exp}_param_{trial_num}_1"

    cfg["exp_name"] = new_exp

    return cfg


def write_trial_config_yaml(
    cfg: Dict[str, Any],
    trial_index: int,
) -> str:
    """
    trial用cfgを、trial専用の出力ディレクトリに
    naf_conf_trial_<trial_index>.yml として保存。

    戻り値: そのyamlの絶対パス。
    """

    exp_dir = os.path.join(cfg["save_loc"], cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    yaml_path = os.path.join(exp_dir, f"naf_conf_trial_{trial_index}.yml")
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    return yaml_path


def run_training_once(
    cfg: Dict[str, Any],
    naf_root: str,
    train_script_path: str,
) -> int:
    """
    1 trial分のNAF学習を subprocess.run() で train.py を直接叩いて実行する。

    train_script_path:
        NAF_extended/model_pipeline/train/train.py のパス（絶対 or 相対）
    naf_root:
        NAF_extended のルートディレクトリ。model_pipeline などをimportできるように
        PYTHONPATHに通す。

    cfg:
        このtrial用の設定dict。これをtrain.pyのCLI引数に展開する。
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
        "--batch_norm", cfg["batch_norm"],
        "--activation_func_name", cfg["activation_func_name"],
        "--save_loc", cfg["save_loc"],
        "--pixel_count", str(cfg["pixel_count"]),
    ]

    print(f"[Optuna] Running training subprocess:\n  {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def gather_best_doa(exp_dir: str) -> float:
    """
    exp_dir/loss_values/loss_epoch_XXXXX.pkl をすべて読み、
    "DoA_err" の最小値を返す（小さいほど良い）。
    1つも読めなければ 999.0 を返す。
    """

    loss_dir = os.path.join(exp_dir, "loss_values")
    if not os.path.isdir(loss_dir):
        print(f"[Optuna] WARNING: loss_values not found at {loss_dir}")
        return 999.0

    pkl_files = sorted(glob.glob(os.path.join(loss_dir, "loss_epoch_*.pkl")))
    if len(pkl_files) == 0:
        print(f"[Optuna] WARNING: no loss_epoch_*.pkl in {loss_dir}")
        return 999.0

    doa_vals = []
    for p in pkl_files:
        try:
            with open(p, "rb") as f:
                record = pickle.load(f)
            if "DoA_err" in record:
                doa_vals.append(float(record["DoA_err"]))
        except Exception as e:
            print(f"[Optuna] ERROR reading {p}: {e}")

    if len(doa_vals) == 0:
        print("[Optuna] WARNING: DoA_err not found in any loss_epoch_*.pkl")
        return 999.0

    return min(doa_vals)


def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna用objective:
      1. base_config -> trial用cfgを生成 (ハイパラとexp_nameを更新)
      2. <save_loc>/<exp_name>/naf_conf_trial_<trial>.yml を保存
      3. train.pyを直接subprocessで走らせる
      4. 学習後、loss_values/*.pkl から最小DoA_errを拾って返す
    """
    global base_config, args

    trial_index = trial.number

    # 1. trial用設定生成
    cfg_trial = make_trial_config(
        base_config,
        base_start_index=args.start_index,
        trial_index=trial_index,
        trial=trial,
    )

    # 2. trial設定yamlを出力ディレクトリに保存
    _trial_yaml_path = write_trial_config_yaml(
        cfg_trial,
        trial_index=trial_index,
    )

    # 3. train.pyを直接実行
    ret = run_training_once(
        cfg=cfg_trial,
        naf_root=args.naf_root,
        train_script_path=args.train_script,
    )

    if ret != 0:
        print(f"[Optuna] WARNING: training subprocess exited with code {ret}")

    # 4. DoA誤差の最小値を読む
    exp_dir = os.path.join(cfg_trial["save_loc"], cfg_trial["exp_name"])
    best_doa_err = gather_best_doa(exp_dir)

    print(f"[Optuna] Trial {trial_index} ({cfg_trial['exp_name']}) -> best DoA_err = {best_doa_err}")
    return best_doa_err


def main():
    parser = argparse.ArgumentParser()

    # ベースとなるNAF設定YAML（固定ハイパラやパスなどを含む）
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="ベースのNAF設定YAML",
    )

    # AVRと同じ param_xxx_1 命名用の開始index
    parser.add_argument(
        "--start_index",
        type=int,
        required=True,
        help="exp_nameに付与する param_{start_index+trial}_1 の 'start_index'",
    )

    # 試行回数
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
    )

    # Optuna Study永続化情報
    parser.add_argument(
        "--study_name",
        type=str,
        default="naf_optuna_study",
        help="Optuna study名（再開にも使う）",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///./optuna_naf.db",
        help="Optuna storage URL (例: sqlite:///./optuna_naf.db)",
    )

    # train.py の場所（直接叩く）
    parser.add_argument(
        "--train_script",
        type=str,
        required=True,
        help="NAFのtrain.pyへのパス (例: ~/NAF_extended/model_pipeline/train/train.py)",
    )

    # PYTHONPATHに追加するルート (model_pipeline がimportできるトップ)
    parser.add_argument(
        "--naf_root",
        type=str,
        default="NAF_extended",
        help="PYTHONPATHに追加するディレクトリ (例: ~/NAF_extended)",
    )

    args_parsed = parser.parse_args()

    # "~" をホームディレクトリに展開
    args_parsed.config = os.path.expanduser(args_parsed.config)
    args_parsed.train_script = os.path.expanduser(args_parsed.train_script)
    args_parsed.naf_root = os.path.expanduser(args_parsed.naf_root)

    # グローバル保持（objectiveから参照する）
    global args
    args = args_parsed

    # base_config のロード
    with open(args.config, "r") as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)

    global base_config
    base_config = loaded

    # Optuna Study 作成 / 再開
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",  # DoA誤差を小さくしたい
    )

    # 最適化開始
    study.optimize(objective, n_trials=args.n_trials)

    # 結果ダンプ
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    print("Study name:", study.study_name)
    print("Storage:", args.storage)


if __name__ == "__main__":
    main()

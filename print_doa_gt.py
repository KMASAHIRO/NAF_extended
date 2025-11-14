#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa
import pyroomacoustics as pra


def angular_error_deg(est_deg: float, true_deg: float) -> float:
    """
    角度差 [deg] を [-180, 180) にラップして絶対値を返す。
    """
    diff = (est_deg - true_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


def load_points(points_path: Path):
    """
    points.txt を読み込み、id(int) -> (x,y,z) の dict を返す。
    """
    id_to_xyz = {}
    with open(points_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid_str, x_str, y_str, z_str = line.split()
            pid = int(pid_str)
            x = float(x_str)
            y = float(y_str)
            z = float(z_str)
            id_to_xyz[pid] = (x, y, z)
    return id_to_xyz


def stft_multi_channel(y: np.ndarray, n_fft: int) -> np.ndarray:
    """
    多チャネル信号 (M, Nt) に対して STFT を計算し (M, F, T) を返す。
    librosa の multi-channel 対応を利用。
    """
    X = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window="hann",
        center=True,
    )
    # librosa.stft は (M, F, T) を返す（M=チャンネル数）
    return X.astype(np.complex64)


def evaluate_doa_from_wav_dataset(
    dataset_root: str,
    fs: int = 16000,
    n_fft: int = 512,
    mic_radius: float = 0.0365,
    algo_names=None,
    subset: str = "all",  # "all", "train", "test"
):
    """
    convert_centered_npz_to_wav_and_points で作成したデータから DoA を推定し，
    物理的真方向との誤差の平均・標準偏差を計算する。

    Parameters
    ----------
    dataset_root : str
        convert_centered_npz_to_wav_and_points の output_root
    fs : int
        サンプリングレート
    n_fft : int
        STFT の FFT 長
    mic_radius : float
        円形アレイの半径 [m]
    algo_names : list[str]
        使用する DOA アルゴリズム名（pyroomacoustics.doa.algorithms のキー）
    subset : {"all", "train", "test"}
        complete.pkl がある場合の評価対象
    """
    if algo_names is None:
        # 必要に応じて増やして OK
        algo_names = ["NormMUSIC"]

    root = Path(dataset_root)
    raw_dir = root / "raw"
    points_path = root / "points.txt"
    split_pkl = root / "train_test_split" / "complete.pkl"

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")
    if not points_path.is_file():
        raise FileNotFoundError(f"points.txt not found: {points_path}")

    # --- 1) points.txt 読み込み: id -> xyz ---
    id_to_xyz = load_points(points_path)

    # --- 2) train/test split の読み込み（あれば） ---
    train_ids = None
    test_ids = None
    if split_pkl.is_file():
        with open(split_pkl, "rb") as f:
            train_ids, test_ids = pickle.load(f)
        train_ids = set(train_ids)
        test_ids = set(test_ids)
        print(f"[Split] complete.pkl 読み込み済: train={len(train_ids)}, test={len(test_ids)}")
    else:
        print("[Split] complete.pkl が見つからないので全サンプルを 'all' として扱います。")
        subset = "all"  # 強制

    # --- 3) wav ファイルを pos_id=source_target ごとにグルーピング ---
    # ファイル名: {source_id}_{target_id}_{ch}.wav
    groups = defaultdict(list)  # pos_id(str) -> list[(ch_idx, wav_path)]

    for wav_path in sorted(raw_dir.glob("*.wav")):
        stem = wav_path.stem  # "source_target_ch"
        try:
            base, ch_str = stem.rsplit("_", 1)
            ch_idx = int(ch_str)
        except ValueError:
            # 想定外のファイル名はスキップ
            continue
        groups[base].append((ch_idx, wav_path))

    # subset フィルタリング
    if subset == "train" and train_ids is not None:
        target_pos_ids = train_ids
    elif subset == "test" and test_ids is not None:
        target_pos_ids = test_ids
    else:
        # "all" or splitなし
        target_pos_ids = set(groups.keys())

    # アルゴごとの誤差を格納
    errors = {algo: [] for algo in algo_names}

    # --- 4) 各 pos_id ごとに DoA 推定 ---
    for pos_id, ch_list in groups.items():
        if pos_id not in target_pos_ids:
            continue

        # pos_id は "sourceID_targetID"
        try:
            source_id_str, target_id_str = pos_id.split("_")
            source_id = int(source_id_str)
            target_id = int(target_id_str)
        except ValueError:
            # 想定外の pos_id 形式
            continue

        if source_id not in id_to_xyz or target_id not in id_to_xyz:
            # 座標情報がなければスキップ
            continue

        src_xyz = np.array(id_to_xyz[source_id], dtype=float)  # (3,)
        tgt_xyz = np.array(id_to_xyz[target_id], dtype=float)  # (3,)

        # 真の方位角（x,y 平面）: ターゲット中心 → ソース方向
        dx = src_xyz[0] - tgt_xyz[0]
        dy = src_xyz[1] - tgt_xyz[1]
        true_rad = math.atan2(dy, dx)
        true_deg = (np.degrees(true_rad) + 360.0) % 360.0

        # チャンネル順に並べて読み込み
        ch_list_sorted = sorted(ch_list, key=lambda x: x[0])
        waveforms = []
        for ch_idx, wpath in ch_list_sorted:
            sig, sr = sf.read(wpath)
            if sr != fs:
                raise ValueError(f"Sampling rate mismatch in {wpath}: {sr} != {fs}")
            if sig.ndim > 1:
                sig = sig[:, 0]  # 万一ステレオなら1ch目だけ使用
            waveforms.append(sig.astype(np.float32))

        # 長さ揃え（念のため）
        min_len = min(len(w) for w in waveforms)
        waveforms = [w[:min_len] for w in waveforms]

        signals = np.stack(waveforms, axis=0)  # (M, Nt)
        M = signals.shape[0]
        if M < 2:
            # DOAできないのでスキップ
            continue

        # マイクアレイの中心は target の (x,y)
        mic_center = tgt_xyz[:2]
        mic_array = pra.beamforming.circular_2D_array(
            center=mic_center,
            M=M,
            radius=mic_radius,
            phi0=np.pi / 2.0,
        )

        # STFT: (M, F, T)
        X = stft_multi_channel(signals, n_fft=n_fft)

        # --- 5) アルゴごとに DoA 推定 & 誤差 ---
        for algo in algo_names:
            try:
                doa = pra.doa.algorithms[algo](mic_array, fs=fs, nfft=n_fft)
                doa.locate_sources(X)

                # 推定方位角（rad）→ deg
                # pyroomacoustics の DOA オブジェクトは azimuth_recon に推定結果を持つ
                # ref: "Spatial spectrum stored in doa.grid.values and estimated directions in doa.azimuth_recon"
                # （pyroomacoustics 論文資料より）
                if doa.azimuth_recon is None or len(doa.azimuth_recon) == 0:
                    continue
                est_rad = doa.azimuth_recon[0]
                est_deg = (np.degrees(est_rad) + 360.0) % 360.0

                err = angular_error_deg(est_deg, true_deg)
                errors[algo].append(err)

            except Exception as e:
                # 失敗したサンプルは無視（必要ならログ出力）
                # print(f"[WARN] algo={algo}, pos_id={pos_id}, error={e}")
                continue

    # --- 6) 結果まとめて表示 ---
    stats = {}
    print("=== DOA Evaluation Results (deg) ===")
    for algo in algo_names:
        vals = np.array(errors[algo], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            print(f"[{algo}] 有効サンプルなし")
            stats[algo] = {"mean": None, "std": None, "n": 0}
            continue
        mean_err = float(np.mean(vals))
        std_err = float(np.std(vals))
        n = int(len(vals))
        print(f"[{algo}] N={n:4d}, mean={mean_err:7.3f} deg, std={std_err:7.3f} deg")
        stats[algo] = {"mean": mean_err, "std": std_err, "n": n}

    return stats


if __name__ == "__main__":
    # ここはあなたのパスに合わせて変更
    #dataset_root = "/home/ach17616qc/Pyroomacoustics/outputs/real_exp_8720_centered_NAF"
    dataset_root = "/home/ach17616qc/AcoustiX/custom_scene/real_env_Smooth_concrete_painted/real_env_Smooth_concrete_painted_centered_standard_NAF"
    #dataset_root = "/home/ach17616qc/Pyroomacoustics/outputs/real_env_avr_16kHz_centered_NAF"

    stats = evaluate_doa_from_wav_dataset(
        dataset_root=dataset_root,
        fs=16000,
        n_fft=512,
        mic_radius=0.0365,
        algo_names=["NormMUSIC"],  # 必要に増やす
        subset="all",              # "train" / "test" も可（complete.pkl があれば）
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAF DoA Evaluation Pipeline
---------------------------------------------
IR(複素STFT) → iSTFT → 白色ノイズ畳み込み → STFT → DoA解析

設計思想:
  - ロード時に全て正規化 (S,G,F,T), (S,2), (S,G,2)
  - 不正shapeや欠損は即エラー（ゼロ埋め禁止）
  - reshape禁止。group_size単位でスライス抽出。
  - 後段処理ではif文・例外処理一切なし。
---------------------------------------------
"""

import os, math, glob, yaml, argparse, pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import librosa
import pyroomacoustics as pra


# ==========================================================
# Config定義とロード
# ==========================================================

@dataclass
class Config:
    # 入力
    npz: Optional[str] = None         # 単一ファイル
    npz_dir: Optional[str] = None     # 複数ファイルディレクトリ
    # 共通設定
    fs: int = 16000
    seeds: List[int] = None
    long_noise_seconds: float = 100.0
    stft_grid: List[Dict[str, Any]] = None  # {nfft, hop, win}
    T_use_list: List[int] = None
    outdir: str = ""
    algo_name: str = "NormMUSIC"
    mic_radius: float = 0.0365
    force: bool = False
    group_size: int = 8               # 追加（既定8）

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # 既定値補完
    raw.setdefault("fs", 16000)
    raw.setdefault("seeds", [0])
    raw.setdefault("long_noise_seconds", 100.0)
    raw.setdefault("stft_grid", [dict(nfft=512, hop=128, win="hann")])
    raw.setdefault("T_use_list", [32])
    raw.setdefault("outdir", "outputs")
    raw.setdefault("algo_name", "NormMUSIC")
    raw.setdefault("mic_radius", 0.0365)
    raw.setdefault("force", False)
    raw.setdefault("group_size", 8)

    return Config(**raw)


# ==========================================================
# 基本ユーティリティ
# ==========================================================

def wrap_0_360(x: float) -> float:
    return float(np.mod(x, 360.0))

def angular_error_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)

def circ_mean_deg(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return float('nan')
    a = np.deg2rad(arr)
    C, S = np.cos(a).mean(), np.sin(a).mean()
    if C == 0.0 and S == 0.0:
        return float('nan')
    return wrap_0_360(np.rad2deg(np.arctan2(S, C)))


# ==========================================================
# 信号処理関連
# ==========================================================

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

def convolve_all(ir: np.ndarray, x: np.ndarray) -> np.ndarray:
    ys = [np.convolve(x, ir[m], mode="full") for m in range(ir.shape[0])]
    return np.stack(ys, axis=0).astype(np.float32)

def stft_full(y: np.ndarray, nfft: int, hop: int, win: str) -> np.ndarray:
    """(G, Nt) → (G, F, T)"""
    win_name = "boxcar" if win in ("none", "rect", "") else win
    X = librosa.stft(
        y,
        n_fft=nfft,
        hop_length=hop,
        window=win_name,
        center=True,
    )
    return X.astype(np.complex64)

def white_noise(L_sec: float, fs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(round(L_sec * fs))).astype(np.float32)


# ==========================================================
# DoA関連
# ==========================================================

def true_angle(tx: np.ndarray, center_xy: np.ndarray) -> float:
    """tx: (2,) 正規化済み"""
    dx, dy = tx[0] - center_xy[0], tx[1] - center_xy[1]
    return wrap_0_360(math.degrees(math.atan2(dy, dx)))

def doa_sliding(X: np.ndarray, fs: int, nfft: int, mic, algo: str, T_use: int):
    """(G,F,T) → (angles_deg[N], centers_idx[N])"""
    T = X.shape[-1]
    if T < T_use:
        return np.array([]), np.array([], dtype=int)
    doa = pra.doa.algorithms[algo](mic, fs=fs, nfft=nfft)

    angs, centers = [], []
    for t0 in range(0, T - T_use + 1, T_use):  # hopを増やすなら第5引数にhopを追加して置換
        Xseg = X[:, :, t0:t0 + T_use]
        doa.locate_sources(Xseg)
        angs.append(float(np.argmax(doa.grid.values)))
        centers.append(t0 + T_use // 2)

    return np.asarray(angs, dtype=float), np.asarray(centers, dtype=int)

# ==========================================================
# 正規化（reshape禁止、欠損は即エラー）
# ==========================================================

def load_and_normalize_npz(cfg: Config):
    G = cfg.group_size
    paths = []

    if cfg.npz_dir:
        paths = sorted(glob.glob(os.path.join(cfg.npz_dir, "*.npz")))
    elif cfg.npz:
        paths = [cfg.npz]
    else:
        raise ValueError("Either npz or npz_dir must be specified")

    all_pred, all_ori, all_tx, all_rx = [], [], [], []

    for path in paths:
        d = np.load(path)
        pred = np.asarray(d["pred_sig_spec"])
        ori  = np.asarray(d["ori_sig_spec"]) if "ori_sig_spec" in d else None
        tx   = np.asarray(d["position_tx"]) if "position_tx" in d else None
        rx   = np.asarray(d["position_rx"]) if "position_rx" in d else None

        if pred.ndim == 3:
            pred = pred[None, ...]
            if ori is not None:
                ori = ori[None, ...]

        S, M, F, T = pred.shape
        if M % G != 0:
            raise ValueError(f"{path}: M={M} not divisible by group_size={G}")
        n_blocks = M // G

        if tx is None:
            raise ValueError(f"{path}: position_tx missing")
        if rx is None:
            raise ValueError(f"{path}: position_rx missing")

        for s in range(S):
            for b in range(n_blocks):
                st, ed = b * G, (b + 1) * G
                all_pred.append(pred[s, st:ed])
                if ori is not None:
                    all_ori.append(ori[s, st:ed])

                # tx: (2,)
                if tx.ndim == 1 and tx.shape[0] == 2:
                    all_tx.append(tx.astype(float))
                elif tx.ndim == 2 and tx.shape == (S, 2):
                    all_tx.append(tx[s])
                elif tx.ndim == 2 and tx.shape == (M, 2):
                    all_tx.append(tx[st])
                elif tx.ndim == 2 and tx.shape == (n_blocks, 2):
                    all_tx.append(tx[b])
                else:
                    raise ValueError(f"{path}: invalid position_tx shape {tx.shape}")

                # rx: (G,2)
                if rx.ndim == 3 and rx.shape == (S, M, 2):
                    all_rx.append(rx[s, st:ed].mean(axis=0))
                elif rx.ndim == 2 and rx.shape == (S, 2):
                    all_rx.append(rx[s])
                elif rx.ndim == 2 and rx.shape == (M, 2):
                    all_rx.append(rx[st:ed].mean(axis=0))
                elif rx.ndim == 2 and rx.shape == (n_blocks, 2):
                    all_rx.append(rx[b])
                else:
                    raise ValueError(f"{path}: invalid position_rx shape {rx.shape}")

    pred_out = np.stack(all_pred).astype(np.complex64)
    ori_out  = np.stack(all_ori).astype(np.complex64) if all_ori else None
    tx_out   = np.stack(all_tx).astype(np.float32)
    rx_out   = np.stack(all_rx).astype(np.float32)

    return dict(pred=pred_out, ori=ori_out, pos_tx=tx_out, pos_rx=rx_out)


# ==========================================================
# グループ単位評価
# ==========================================================

def process_group(spec, tx, rx, fs, stft_cfg, algo, radius, T_use, x_long):
    # 1) (G,F,T) -> (G,Nt)
    ir = istft_ir(spec)  # あなたの既定（nfft=512, hop=128, hann）

    # 2) 長尺ノイズと畳み込み -> (G, T_full)
    y = convolve_all(ir, x_long)

    # 3) 全長STFT -> (G, F, Tfrm)
    X = stft_full(y, stft_cfg["nfft"], stft_cfg["hop"], stft_cfg["win"])

    # 4) 真値角（tx -> rx）
    true_deg = true_angle(tx, rx)

    # 5) マイク配列（M は IR のチャンネル数）
    M = ir.shape[0]
    mic = pra.beamforming.circular_2D_array(center=rx, M=M, radius=radius, phi0=math.pi/2)

    # 6) スライディングDoA（角度列 + centers）
    angs, centers = doa_sliding(X, fs, stft_cfg["nfft"], mic, algo, T_use)

    # 7) 集計
    mu  = circ_mean_deg(angs) if len(angs) else np.nan
    med = float(np.median(angs)) if len(angs) else np.nan

    return dict(
        true_deg=float(true_deg),
        mu_deg=float(mu) if np.isfinite(mu) else np.nan,
        med_deg=float(med) if np.isfinite(med) else np.nan,
        err_mu=float(angular_error_deg(mu, true_deg)) if np.isfinite(mu) else np.nan,
        err_med=float(angular_error_deg(med, true_deg)) if np.isfinite(med) else np.nan,
        angles_deg=angs.tolist(),
        centers=list(map(int, centers.tolist()))  # ← ここで保持
    )


# ==========================================================
# メインルーチン
# ==========================================================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = load_config(args.cfg)

    # === outdir と config_effective.yaml 保存 ===
    root = os.path.expanduser(cfg.outdir)
    _ensure_dir(root)
    with open(os.path.join(root, "config_effective.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.__dict__, f, allow_unicode=True, sort_keys=False)

    # ---- 正規化済みデータをロード ----
    data = load_and_normalize_npz(cfg)

    results = []

    # ---- 全 seed × stft_grid × T_use を総当たり ----
    for seed in cfg.seeds:
        x_long = white_noise(cfg.long_noise_seconds, cfg.fs, seed)

        for stft_cfg in cfg.stft_grid:
            nfft = int(stft_cfg["nfft"])
            hop  = int(stft_cfg["hop"])
            win  = str(stft_cfg["win"])

            # 条件ディレクトリ（例：stft_hann_L512_H128）
            stft_tag = f"stft_{(win or 'none').lower()}_L{nfft}_H{hop}"
            stft_dir = os.path.join(root, stft_tag)
            _ensure_dir(stft_dir)

            for T_use in cfg.T_use_list:
                # T_use ディレクトリ & pkl パス
                tdir = os.path.join(stft_dir, f"Tuse_{int(T_use)}")
                _ensure_dir(tdir)
                pkl_path = os.path.join(tdir, "results.pkl")

                # ---- force チェック：既存 pkl があればロードしてCSV集計だけ行う ----
                if os.path.exists(pkl_path) and not cfg.force:
                    print(f"[LOAD] {pkl_path} already exists → reuse for CSV only")
                    with open(pkl_path, "rb") as f:
                        saved = pickle.load(f)
                    # entries_this_condition にロード内容を格納
                    entries_this_condition = saved.get("entries", [])
                
                    # 既存pklの内容からCSV行を生成
                    for rec_entry in entries_this_condition:
                        rec_pred = rec_entry.get("pred", {})
                        rec_ori  = rec_entry.get("ori", {})
                        err_mu_vs_ori  = rec_entry.get("err_mu_vs_ori", np.nan)
                        err_med_vs_ori = rec_entry.get("err_med_vs_ori", np.nan)
                
                        rows_flat = dict(
                            index=int(rec_entry.get("index", -1)),
                            seed=int(rec_entry.get("seed", -1)),
                            nfft=int(rec_entry.get("nfft", -1)),
                            hop=int(rec_entry.get("hop", -1)),
                            win=rec_entry.get("win", ""),
                            T_use=int(rec_entry.get("T_use", -1)),
                            pred_mu_deg=rec_pred.get("mu_deg", np.nan),
                            pred_med_deg=rec_pred.get("med_deg", np.nan),
                            pred_err_mu=rec_pred.get("err_mu", np.nan),
                            pred_err_med=rec_pred.get("err_med", np.nan),
                            ori_mu_deg=rec_ori.get("mu_deg", np.nan) if rec_ori else np.nan,
                            ori_med_deg=rec_ori.get("med_deg", np.nan) if rec_ori else np.nan,
                            ori_err_mu=rec_ori.get("err_mu", np.nan) if rec_ori else np.nan,
                            ori_err_med=rec_ori.get("err_med", np.nan) if rec_ori else np.nan,
                            err_mu_vs_ori=err_mu_vs_ori,
                            err_med_vs_ori=err_med_vs_ori,
                        )
                        results.append(rows_flat)
                
                    # 既存結果のCSV反映が済んだらスキップ（再計算しない）
                    continue

                # この条件の全グループ結果をためる箱（pkl用）
                entries_this_condition = []

                for s in range(data["pred"].shape[0]):
                    spec_pred = data["pred"][s]
                    spec_ori  = data["ori"][s] if data["ori"] is not None else None
                    tx = data["pos_tx"][s]
                    rx = data["pos_rx"][s]

                    rec_pred = process_group(
                        spec_pred, tx, rx,
                        cfg.fs,
                        dict(nfft=nfft, hop=hop, win=win),
                        cfg.algo_name,
                        cfg.mic_radius,
                        T_use,
                        x_long
                    )

                    rec_ori = None
                    if spec_ori is not None:
                        rec_ori = process_group(
                            spec_ori, tx, rx,
                            cfg.fs,
                            dict(nfft=nfft, hop=hop, win=win),
                            cfg.algo_name,
                            cfg.mic_radius,
                            T_use,
                            x_long
                        )
                        err_mu_vs_ori  = angular_error_deg(rec_pred["mu_deg"],  rec_ori["mu_deg"])  if np.isfinite(rec_pred["mu_deg"])  and np.isfinite(rec_ori["mu_deg"])  else np.nan
                        err_med_vs_ori = angular_error_deg(rec_pred["med_deg"], rec_ori["med_deg"]) if np.isfinite(rec_pred["med_deg"]) and np.isfinite(rec_ori["med_deg"]) else np.nan
                    else:
                        err_mu_vs_ori, err_med_vs_ori = np.nan, np.nan

                    # --- pkl 用：pred / ori のネスト構造のみ保持 ---
                    rec_entry = dict(
                        pred=rec_pred,
                        ori=rec_ori,
                        index=int(s),
                        seed=int(seed),
                        nfft=int(nfft),
                        hop=int(hop),
                        win=win,
                        T_use=int(T_use),
                    )

                    # --- CSV 用：pred_*/ori_*列で明確化 ---
                    rows_flat = dict(
                        index=int(s), seed=int(seed),
                        nfft=int(nfft), hop=int(hop), win=win, T_use=int(T_use),
                        pred_mu_deg=(float(rec_pred["mu_deg"]) if np.isfinite(rec_pred["mu_deg"]) else np.nan),
                        pred_med_deg=(float(rec_pred["med_deg"]) if np.isfinite(rec_pred["med_deg"]) else np.nan),
                        pred_err_mu=(float(rec_pred["err_mu"]) if np.isfinite(rec_pred["err_mu"]) else np.nan),
                        pred_err_med=(float(rec_pred["err_med"]) if np.isfinite(rec_pred["err_med"]) else np.nan),
                        ori_mu_deg=(float(rec_ori["mu_deg"]) if rec_ori and np.isfinite(rec_ori["mu_deg"]) else np.nan),
                        ori_med_deg=(float(rec_ori["med_deg"]) if rec_ori and np.isfinite(rec_ori["med_deg"]) else np.nan),
                        ori_err_mu=(float(rec_ori["err_mu"]) if rec_ori and np.isfinite(rec_ori["err_mu"]) else np.nan),
                        ori_err_med=(float(rec_ori["err_med"]) if rec_ori and np.isfinite(rec_ori["err_med"]) else np.nan),
                        err_mu_vs_ori=(float(err_mu_vs_ori) if np.isfinite(err_mu_vs_ori) else np.nan),
                        err_med_vs_ori=(float(err_med_vs_ori) if np.isfinite(err_med_vs_ori) else np.nan),
                    )

                    results.append(rows_flat)
                    entries_this_condition.append(rec_entry)
                    print(f"[OK] seed={seed} nfft={nfft} hop={hop} T={T_use} group={s+1}/{data['pred'].shape[0]}")


                # === 条件ごと pkl 保存（meta 付き） ===
                with open(pkl_path, "wb") as f:
                    pickle.dump(dict(
                        meta=dict(
                            condition=stft_tag,
                            Tuse=int(T_use),
                            fs=int(cfg.fs),
                            long_noise_seconds=float(cfg.long_noise_seconds),
                            stft_win=(win or 'none').lower(),
                            stft_nfft=int(nfft),
                            stft_hop=int(hop),
                            seeds=list(cfg.seeds),
                            algo=str(cfg.algo_name),
                            mic_radius=float(cfg.mic_radius),
                        ),
                        entries=entries_this_condition
                    ), f)
                print("[OK] saved:", pkl_path)

    # ---- まとめ CSV ----
    df = pd.DataFrame(results)
    csv_path = os.path.join(root, "summary.csv")
    df.to_csv(csv_path, index=False)
    print("[DONE] summary saved →", csv_path)


if __name__ == "__main__":
    main()

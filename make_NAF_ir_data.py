import os
import numpy as np
import soundfile as sf
import pickle
from pathlib import Path
from tqdm import tqdm
import random

def convert_npz_to_wav_and_points(
    npz_root: str,
    output_root: str,
    sampling_rate: int = 16000,
    ir_len: int = 1600,
    room_dim: tuple = (6.110, 8.807, 2.7),
):
    npz_root = Path(npz_root)
    output_root = Path(output_root)
    raw_dir = output_root / "raw"
    points_path = output_root / "points.txt"
    minmax_path = output_root / "minmax" / "minmax.pkl"
    split_dir = output_root / "train_test_split"

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(minmax_path.parent, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    all_points = []
    point_id_counter = 0
    point_id_map = {}
    pos_id_set = set()
    tx_rx_ch_to_pos_id = dict()

    for tx_dir in sorted(npz_root.glob("tx_*"), key=lambda p: int(p.name.split("_")[1])):
        tx_idx = int(tx_dir.name.split("_")[1])
        for rx_dir in sorted(tx_dir.glob("rx_*"), key=lambda p: int(p.name.split("_")[1])):
            rx_idx = int(rx_dir.name.split("_")[1])
            ir_files = sorted(rx_dir.glob("*.npz"))
            if len(ir_files) == 0:
                continue

            for ir_file in ir_files:
                ir_idx = int(ir_file.name.split("_")[1].split(".")[0])
                data = np.load(ir_file)
                ir = data["ir"][:ir_len]
                ch_idx = 1  # 明示的な ch_idx がないため 1 に固定
                position_tx = tuple(np.round(data["position_tx"], 6))
                position_rx = tuple(np.round(data["position_rx"], 6))

                if position_tx not in point_id_map:
                    point_id_map[position_tx] = point_id_counter
                    all_points.append((point_id_counter, *position_tx))
                    point_id_counter += 1
                source_id = point_id_map[position_tx]

                if position_rx not in point_id_map:
                    point_id_map[position_rx] = point_id_counter
                    all_points.append((point_id_counter, *position_rx))
                    point_id_counter += 1
                target_id = point_id_map[position_rx]

                pos_id = f"{source_id}_{target_id}"
                pos_id_set.add(pos_id)
                tx_rx_ch_to_pos_id[f"{tx_idx}_{rx_idx}_{ir_idx}"] = pos_id

                wav_path = raw_dir / f"{pos_id}_{ch_idx}.wav"
                sf.write(wav_path, ir, sampling_rate)

    # === points.txt ===
    with open(points_path, "w") as f:
        for pid, x, y, z in sorted(all_points, key=lambda x: x[0]):
            f.write(f"{pid}\t{x:.6f}\t{y:.6f}\t{z:.6f}\n")

    # === minmax.pkl ===
    min_xyz = np.array([0, 0, 0], dtype=np.float32)
    max_xyz = np.array(room_dim, dtype=np.float32)
    with open(minmax_path, "wb") as f:
        pickle.dump((min_xyz, max_xyz), f)

    # === train_test_split.pkl に従って complete.pkl を生成 ===
    train_test_split_pkl = npz_root / "train_test_split.pkl"
    if train_test_split_pkl.exists():
        with open(train_test_split_pkl, "rb") as f:
            split_data = pickle.load(f)

        def extract_pos_ids(relpaths):
            pos_ids = list()
            for p in relpaths:
                parts = Path(p).parts  # tx_*/rx_*/ir_*.npz
                try:
                    tx_idx = int(parts[0].split("_")[1])
                    rx_idx = int(parts[1].split("_")[1])
                    ir_idx = int(parts[2].split("_")[1].split(".")[0])
                    key = f"{tx_idx}_{rx_idx}_{ir_idx}"
                    if tx_rx_ch_to_pos_id[key] not in pos_ids:
                        pos_ids.append(tx_rx_ch_to_pos_id[key])
                except (IndexError, ValueError):
                    continue
            return pos_ids

        train_ids = extract_pos_ids(split_data.get("train", []))
        test_ids = extract_pos_ids(split_data.get("test", []))

        complete_split = [train_ids, test_ids]
        with open(split_dir / "complete.pkl", "wb") as f:
            pickle.dump(complete_split, f)

        print(f"[Split] train_test_split.pkl → complete.pkl 作成済")
        print(f"[Split] train={len(train_ids)}, test={len(test_ids)}")
    else:
        print(f"Warning: train_test_split.pkl not found. Skipping complete.pkl creation.")

    print(f"✅ 完了: 総ポイント数 {len(all_points)}")
    print(f"📁 出力先: {output_root}")

# === 実行例 ===
if __name__ == "__main__":
    convert_npz_to_wav_and_points(
        npz_root="/home/ach17616qc/Pyroomacoustics/outputs/real_env_avr_16kHz",
        output_root="/home/ach17616qc/Pyroomacoustics/outputs/real_env_avr_16kHz_NAF"
    )

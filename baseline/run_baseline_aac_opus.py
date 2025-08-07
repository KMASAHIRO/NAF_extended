import os
import subprocess
import argparse
import shutil
import pickle
from joblib import Parallel, delayed
import numpy as np
from scipy.io.wavfile import read
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from time import time

# ========== AAC処理 ==========
def process_aac(ffmpeg, inp, temp_m4a, temp_wav, final_wav):
    try:
        encode_cmd = f"{ffmpeg} -i {inp} -c:a aac -b:a 24k {temp_m4a}"
        decode_cmd = f"{ffmpeg} -i {temp_m4a} -c:a pcm_f32le -ar 22050 {temp_wav}"
        subprocess.call(encode_cmd, timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        subprocess.call(decode_cmd, timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        shutil.copyfile(temp_wav, final_wav)
        outsize = os.path.getsize(temp_m4a)
        os.remove(temp_m4a)
        os.remove(temp_wav)
        return outsize
    except Exception as e:
        print(f"AAC processing error: {e}")
        return 0

# ========== Opus処理 ==========
def process_opus(opus_enc, opus_dec, inp, temp_opus, temp_wav, final_wav):
    try:
        encode_cmd = f"{opus_enc} {inp} {temp_opus} --bitrate 6 --music --comp 10 --discard-comments --discard-pictures --cvbr"
        decode_cmd = f"{opus_dec} {temp_opus} {temp_wav} --rate 22050 --float"
        subprocess.call(encode_cmd, timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        subprocess.call(decode_cmd, timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        shutil.copyfile(temp_wav, final_wav)
        outsize = os.path.getsize(temp_opus)
        os.remove(temp_opus)
        os.remove(temp_wav)
        return outsize
    except Exception as e:
        print(f"Opus processing error: {e}")
        return 0

# ========== エンコード共通処理 ==========
def encode_all(raw_path, write_path, ramdisk_path, codec, ffmpeg=None, opus_enc=None, opus_dec=None):
    os.makedirs(write_path, exist_ok=True)
    files = sorted([f for f in os.listdir(raw_path) if f.endswith(".wav")])
    total_container, total_temp, total_out, final_loc = [], [], [], []

    for i, ff in enumerate(files):
        cur_file = os.path.join(raw_path, ff)
        if os.path.getsize(cur_file) < 1024:
            #print(f"Skipping small file: {cur_file}")
            print(f"⚠️ Small file detected: {cur_file}")
            #continue
        if os.path.exists(os.path.join(write_path, ff)):
            continue
        total_container.append(cur_file)
        ext = ".m4a" if codec == "aac" else ".opus"
        total_temp.append(os.path.join(ramdisk_path, f"{i}{ext}"))
        total_out.append(os.path.join(ramdisk_path, f"{i}.wav"))
        final_loc.append(os.path.join(write_path, ff))

    if codec == "aac":
        out = Parallel(n_jobs=16, verbose=5)(delayed(process_aac)(
            ffmpeg, i, t, o, f) for i, t, o, f in zip(total_container, total_temp, total_out, final_loc))
        pkl_name = "sizes_aac.pkl"
    else:
        out = Parallel(n_jobs=16, verbose=5)(delayed(process_opus)(
            opus_enc, opus_dec, i, t, o, f) for i, t, o, f in zip(total_container, total_temp, total_out, final_loc))
        pkl_name = "sizes_opus.pkl"

    with open(os.path.join(write_path, pkl_name), "wb") as writer:
        pickle.dump(out, writer)

# ========== 補間処理 ==========
def run_interpolation(baseline_mode, interp_mode, args):
    print(f"\n=== Running interpolation: {baseline_mode}-{interp_mode} ===")

    # train/test split
    split_path = os.path.join(args.split_loc, "complete.pkl")
    with open(split_path, "rb") as f:
        train_split, test_split = pickle.load(f)

    # 座標読み込み
    coors = np.loadtxt(args.points_path)[:, 1:][:, [0, 1]]
    coors = coors.astype(np.single)

    # フォルダ選択
    if baseline_mode == "opus":
        train_folder = args.opus_write_path
    else:
        train_folder = args.aac_write_path

    os.makedirs(args.result_output_dir, exist_ok=True)
    save_name = os.path.join(args.result_output_dir, f"{baseline_mode}_{interp_mode}.pkl")
    if os.path.isfile(save_name):
        print(f"Result already exists: {save_name}")
        return

    # チャンネル数推定
    example_str = train_split[0]
    channel_count = 0
    while os.path.exists(os.path.join(train_folder, f"{example_str}_{channel_count}.wav")):
        channel_count += 1
    print(f"Detected {channel_count} channels.")

    # 各チャンネルについて補間処理
    interpolators = []
    max_lengths = []
    for ch in range(channel_count):
        print(f"Processing channel {ch}...")

        train_coors = []
        train_data = []

        for train_str in train_split:
            listener, emitter = train_str.split("_")
            total_pos = np.concatenate((coors[int(listener)], coors[int(emitter)]), axis=0)
            wav_path = os.path.join(train_folder, f"{train_str}_{ch}.wav")
            try:
                sr, data = read(wav_path)
            except:
                print(f"Missing {wav_path}")
                continue
            if not np.all(np.isfinite(data)):
                continue
            train_coors.append(total_pos)
            train_data.append(data.astype(np.single))

        if len(train_data) == 0:
            raise RuntimeError(f"No valid training data found for channel {ch}")

        max_len = max([x.shape[0] for x in train_data])
        max_lengths.append(max_len)
        train_data = np.array([
            np.pad(x, (0, max_len - x.shape[0])) for x in train_data
        ])
        train_coors = np.array(train_coors)

        print(f"Training data shape for channel {ch}: {train_data.shape}")
        print(f"Building {interp_mode} interpolator...")

        if interp_mode == "linear":
            interp_engine = LinearNDInterpolator(points=train_coors, values=train_data, fill_value=0.0, rescale=False)
        else:
            interp_engine = NearestNDInterpolator(x=train_coors, y=train_data)

        interpolators.append(interp_engine)

    # テストデータ補間
    container = dict()
    gt_container = dict()
    print("Interpolating test data...")
    for test_str in test_split:
        listener, emitter = test_str.split("_")
        total_pos = np.concatenate((coors[int(listener)], coors[int(emitter)]), axis=0)

        ch_outputs = []
        gt_outputs = []
        for ch in range(channel_count):
            wav_path = os.path.join(train_folder, f"{test_str}_{ch}.wav")
            
            sr, gt_data = read(wav_path)
            
            gt_data = gt_data.astype(np.single)

            # Pad GT to match predicted length
            gt_data = np.pad(gt_data, (0, max_lengths[ch] - len(gt_data)))

            out = interpolators[ch](total_pos)
            if out is None:
                print(f"Interpolation failed at {test_str} ch {ch}")
                out = np.zeros(max_lengths[ch], dtype=np.single)

            ch_outputs.append(out.astype(np.single))
            gt_outputs.append(gt_data)

        container[test_str] = np.stack(ch_outputs, axis=0)  # shape: (channel_count, time)
        gt_container[test_str] = np.stack(gt_outputs, axis=0)
    
    # pred + gt をまとめて保存
    output = {
        "pred": container,
        "gt": gt_container
    }
    
    # 保存
    with open(save_name, "wb") as f:
        pickle.dump(container, f)
    print(f"Results saved: {save_name}")

# ========== メイン ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", required=True)
    parser.add_argument("--aac_write_path", default="./aac_enc_test")
    parser.add_argument("--opus_write_path", default="./opus_enc_test")
    parser.add_argument("--ramdisk_path", default="/mnt/ramdisk")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--opus_enc", default="opusenc")
    parser.add_argument("--opus_dec", default="opusdec")
    parser.add_argument("--split_loc", default="./train_test_split")
    parser.add_argument("--points_path", default="./points.txt")
    parser.add_argument("--result_output_dir", default="./baseline_results")
    args = parser.parse_args()

    # AAC encode/decode
    encode_all(args.raw_path, args.aac_write_path, args.ramdisk_path, codec="aac", ffmpeg=args.ffmpeg)
    # Opus encode/decode
    encode_all(args.raw_path, args.opus_write_path, args.ramdisk_path, codec="opus", opus_enc=args.opus_enc, opus_dec=args.opus_dec)

    # Interpolation for AAC and Opus
    for baseline in ["aac", "opus"]:
        for interp in ["nearest", "linear"]:
            run_interpolation(baseline, interp, args)

    print("\n✅ All baselines processed successfully.")

import numpy as np
from scipy import stats
from scipy.signal import hilbert
import scipy
import auraloss
import torch

import math
import pickle
import pyroomacoustics as pra

def metric_cal(ori_ir, pred_ir, fs=16000, window=32):
    """calculate the evaluation metric

    Parameters
    ----------
    ori_ir : np.array
        ground truth impulse response
    pred_ir : np.array
        predicted impulse response
    fs : int
        sampling rate, by default 16000

    Returns
    -------
    evaluation metrics
    """
    
    if ori_ir.ndim == 1:
        ori_ir = ori_ir[np.newaxis, :]
    if pred_ir.ndim == 1:
        pred_ir = pred_ir[np.newaxis, :]

    # prevent numerical issue for log calculation
    multi_stft = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[512, 256, 128], win_lengths=[300, 150, 75], hop_sizes=[60, 30, 8])
    multi_stft_loss = multi_stft(torch.tensor(ori_ir).unsqueeze(1), torch.tensor(pred_ir).unsqueeze(1))

    fft_ori = np.fft.fft(ori_ir, axis=-1)
    fft_predict = np.fft.fft(pred_ir, axis=-1)

    angle_error = np.mean(np.abs(np.cos(np.angle(fft_ori)) - np.cos(np.angle(fft_predict)))) + np.mean(np.abs(np.sin(np.angle(fft_ori)) - np.sin(np.angle(fft_predict))))
    amp_ori = scipy.ndimage.convolve1d(np.abs(fft_ori), np.ones(window))
    amp_predict = scipy.ndimage.convolve1d(np.abs(fft_predict), np.ones(window))
    amp_error = np.mean(np.abs(amp_ori - amp_predict) / amp_ori)

    # calculate the envelop error
    ori_env = np.abs(hilbert(ori_ir))
    pred_env = np.abs(hilbert(pred_ir))
    env_error = np.mean(np.abs(ori_env - pred_env) / np.max(ori_env, axis=1, keepdims=True))

    # derevie the energy trend
    ori_energy = 10.0 * np.log10(np.cumsum(ori_ir[:,::-1]**2 + 1e-9, axis=-1)[:,::-1])
    pred_energy = 10.0 * np.log10(np.cumsum(pred_ir[:,::-1]**2 + 1e-9, axis=-1)[:,::-1])

    ori_energy -= ori_energy[:, 0].reshape(-1, 1)
    pred_energy -= pred_energy[:, 0].reshape(-1, 1)
    
    # calculate the t60 percentage error and EDT time error
    ori_t60, ori_edt = t60_EDT_cal(ori_energy, fs=fs)
    pred_t60, pred_edt = t60_EDT_cal(pred_energy, fs=fs)
    t60_error = np.mean(np.abs(ori_t60 - pred_t60) / ori_t60)
    edt_error = np.mean(np.abs(ori_edt - pred_edt))

    # calculate the C50 error
    base_sample = 0
    samples_50ms = int(0.05 * fs) + base_sample  # Number of samples in 50 ms
    # Compute the energy in the first 50ms and from 50ms to the end
    energy_ori_early = np.sum(ori_ir[:,base_sample:samples_50ms]**2, axis=-1)
    energy_ori_late = np.sum(ori_ir[:,samples_50ms:]**2, axis=-1)
    energy_pred_early = np.sum(pred_ir[:,base_sample:samples_50ms]**2, axis=-1)
    energy_pred_late = np.sum(pred_ir[:,samples_50ms:]**2, axis=-1)

    # Calculate C50 for the original and predicted impulse response
    C50_ori = 10.0 * np.log10(energy_ori_early / energy_ori_late)
    C50_pred = 10.0 * np.log10(energy_pred_early / energy_pred_late)
    C50_error = np.mean(np.abs(C50_ori - C50_pred))

    return angle_error, amp_error, env_error, t60_error, edt_error, C50_error, multi_stft_loss, ori_energy, pred_energy


def t60_EDT_cal(energys, init_db=-5, end_db=-25, factor=3.0, fs=16000):
    """calculate the T60 and EDT metric of the given impulse response normalized energy trend
    t60: find the time it takes to decay from -5db to -65db.
        A usual way to do this is to calculate the time it takes from -5 to -25db, and multiply by 3.0
    
    EDT: Early decay time, time it takes to decay from 0db to -10db, and multiply the number by 6

    Parameters
    ----------
    energys : np.array
        normalized energy
    init_db : int, optional
        t60 start db, by default -5
    end_db : int, optional
        t60 end db, by default -25
    factor : float, optional
        t60 multiply factor, by default 3.0
    fs : int, optional
        sampling rate, by default 16000

    Returns
    -------
    t60 : float
    edt : float, seconds
    """

    t60_all = []
    edt_all = []

    for energy in energys:
        # find the -10db point
        edt_factor = 6.0
        energy_n10db = energy[np.abs(energy - (-10)).argmin()]

        n10db_sample = np.where(energy == energy_n10db)[0][0]
        edt = n10db_sample / fs * edt_factor

        # find the intersection of -5db and -25db position
        energy_init = energy[np.abs(energy - init_db).argmin()]
        energy_end = energy[np.abs(energy - end_db).argmin()]
        init_sample = np.where(energy == energy_init)[0][0]
        end_sample = np.where(energy == energy_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = energy[init_sample:end_sample + 1]
        
        # regress to find the db decay trend
        slope, intercept = stats.linregress(x, y)[0:2]
        db_regress_init = (init_db - intercept) / slope
        db_regress_end = (end_db - intercept) / slope

        # get t60 value
        t60 = factor * (db_regress_end - db_regress_init)
        
        t60_all.append(t60)
        edt_all.append(edt)

    t60_all = np.array(t60_all)
    edt_all = np.array(edt_all)

    return t60_all, edt_all

def angular_error_deg(est_deg, ref_deg):
    return min(abs(est_deg - ref_deg), 360 - abs(est_deg - ref_deg))

def run_doa_on_pkl(pkl_path, fs=16000, n_fft=512, mic_radius=0.0365,
                   algo_names=None, mic_array_size=8, save_path=None):
    if algo_names is None:
        algo_names = ['MUSIC', 'NormMUSIC', 'SRP', 'CSSM', 'WAVES', 'TOPS', 'FRIDA']

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    keys = sorted(data["pred"].keys())

    # --- 真値DoAを tx_pos と rx_pos から計算 ---
    def _bearing_deg_from_tx_rx(tx, rx):
        dx, dy = tx[0] - rx[0], tx[1] - rx[1]
        deg = math.degrees(math.atan2(dy, dx))
        return deg + 360 if deg < 0 else deg

    true_deg_list = [
        _bearing_deg_from_tx_rx(np.asarray(data["tx_pos"][k]), np.asarray(data["rx_pos"][k]))
        for k in keys
    ]

    # === pred/gt をそのまま時間波形として展開 ===
    pred_all, gt_all = [], []
    for k in keys:
        pred_all.extend(data["pred"][k])  # shape: (ch, time)
        gt_all.extend(data["gt"][k])
    pred_all = np.stack(pred_all)  # shape: (N, T)
    gt_all = np.stack(gt_all)

    N, T = pred_all.shape
    G = N // mic_array_size
    assert N % mic_array_size == 0
    assert G == len(keys)

    doa_results = {algo: {
        "true_deg": [], "pred_deg": [], "gt_deg": [],
        "pred_vs_gt_error": [], "pred_vs_true_error": [], "gt_vs_true_error": []
    } for algo in algo_names}

    for g in range(G):
        idxs = np.arange(g * mic_array_size, (g + 1) * mic_array_size)
        pred_group = pred_all[idxs]
        gt_group = gt_all[idxs]

        # マイクアレイ定義
        mic_array = pra.beamforming.circular_2D_array(
            center=np.array([0.0, 0.0]),
            M=mic_array_size,
            radius=mic_radius,
            phi0=np.pi / 2
        )

        true_deg = true_deg_list[g]

        # STFT変換
        def compute_stft(signals):
            return np.array([
                pra.transform.stft.analysis(sig, n_fft, n_fft // 2)
                for sig in signals
            ])

        X_pred = np.transpose(compute_stft(pred_group), (0, 2, 1))
        X_gt = np.transpose(compute_stft(gt_group), (0, 2, 1))

        for algo in algo_names:
            try:
                doa_pred = pra.doa.algorithms[algo](mic_array, fs=fs, nfft=n_fft)
                doa_pred.locate_sources(X_pred)
                doa_gt = pra.doa.algorithms[algo](mic_array, fs=fs, nfft=n_fft)
                doa_gt.locate_sources(X_gt)

                if algo == 'FRIDA':
                    pred_deg = np.argmax(np.abs(doa_pred._gen_dirty_img()))
                    gt_deg = np.argmax(np.abs(doa_gt._gen_dirty_img()))
                else:
                    pred_deg = np.argmax(doa_pred.grid.values)
                    gt_deg = np.argmax(doa_gt.grid.values)

                err_pred_vs_gt = angular_error_deg(pred_deg, gt_deg)
                err_pred_vs_true = angular_error_deg(pred_deg, true_deg)
                err_gt_vs_true = angular_error_deg(gt_deg, true_deg)

                doa_results[algo]["true_deg"].append(true_deg)
                doa_results[algo]["pred_deg"].append(pred_deg)
                doa_results[algo]["gt_deg"].append(gt_deg)
                doa_results[algo]["pred_vs_gt_error"].append(err_pred_vs_gt)
                doa_results[algo]["pred_vs_true_error"].append(err_pred_vs_true)
                doa_results[algo]["gt_vs_true_error"].append(err_gt_vs_true)

            except Exception as e:
                print(f"DoA failed (group {g}, algo {algo}): {e}")
                for key in doa_results[algo]:
                    doa_results[algo][key].append(None)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(doa_results, f)
        print(f"DoA results saved to {save_path}")

    return doa_results

import pyroomacoustics as pra
import librosa
import numpy as np
import math
import pickle

if __name__ == "__main__":
    mic_radius = 0.1
    mic_num = 4
    fs = 22050
    n_fft = 512
    points_path = "./wav_data/points.txt"
    write_path = "./DoA_train.pkl"

    with open("./results/inference_wav/" + "output_train_NAF.pkl", mode="rb") as f:
        output_wav = pickle.load(f)
    
    with open(points_path, "r") as f:
        lines = f.readlines()
    coords = [x.replace("\n", "").split("\t") for x in lines]

    algo_names = ['MUSIC', 'NormMUSIC', 'SRP', 'CSSM', 'WAVES', 'TOPS', 'FRIDA']
    degree_diff_gt = dict()
    degree_diff_net = dict()
    degree_diff_gt_net = dict()
    doa_values = dict()
    for x in algo_names:
        degree_diff_gt[x] = list()
        degree_diff_net[x] = list()
        degree_diff_gt_net[x] = list()
        doa_values[x] = {"gt":{}, "net":{}}
    
    for key in output_wav.keys():
        node_names = key.replace("[", "").replace("]", "").replace("'", "").split(",")
        source_index = int(node_names[0])
        mic_index = int(node_names[1])

        # Create circular microphone positions from mic_index
        mic_position_str = coords[mic_index]
        mic_position = [float(x) for x in mic_position_str][1:3]
        position_circle_xy = pra.beamforming.circular_2D_array(center=mic_position, M=mic_num, phi0=0, radius=mic_radius)

        # Calculate the correct direction
        source_position_str = coords[source_index]
        source_position = [float(x) for x in source_position_str][1:3]
        radian = math.atan2(source_position[1] - mic_position[1], source_position[0] - mic_position[0])
        if radian < 0:
            radian += 2*math.pi
        degree_coords = radian * (180 / math.pi)

        # DoA calculation (multiple algorithms)
        # Calculate the predicted direction from DoA
        for x in algo_names:
            doa_gt = pra.doa.algorithms[x](position_circle_xy, fs=fs, nfft=n_fft)
            spectrograms_gt = librosa.stft(np.asarray(output_wav[key]["gt_wav"]),n_fft=n_fft, hop_length=n_fft//4)
            doa_gt.locate_sources(spectrograms_gt)
            if x == 'FRIDA':
                gt_degree = np.argmax(np.abs(doa_gt._gen_dirty_img()))
                gt_values = doa_gt._gen_dirty_img()
            else:
                gt_degree = np.argmax(doa_gt.grid.values)
                gt_values = doa_gt.grid.values

            doa_net = pra.doa.algorithms[x](position_circle_xy, fs=fs, nfft=n_fft)
            spectrograms_net = librosa.stft(np.asarray(output_wav[key]["net_wav"]),n_fft=n_fft, hop_length=n_fft//4)
            doa_net.locate_sources(spectrograms_net)
            if x == 'FRIDA':
                net_degree = np.argmax(np.abs(doa_net._gen_dirty_img()))
                net_values = doa_net._gen_dirty_img()
            else:
                net_degree = np.argmax(doa_net.grid.values)
                net_values = doa_net.grid.values
            
            # Calculate the angle (degree) difference between the predicted and correct directions
            degree_diff_gt[x].append(np.abs(gt_degree - degree_coords))
            degree_diff_net[x].append(np.abs(net_degree - degree_coords))
            degree_diff_gt_net[x].append(np.abs(net_degree - gt_degree))
            
            # Save the DoA results
            doa_values[x]["gt"][key] = gt_values
            doa_values[x]["net"][key] = net_values
    
    # Calculate the average of the radian differences
    degree_diff_gt_mean = dict()
    degree_diff_net_mean = dict()
    degree_diff_gt_net_mean = dict()
    for x in algo_names:
        degree_diff_gt_mean[x] = np.mean(degree_diff_gt[x])
        degree_diff_net_mean[x] = np.mean(degree_diff_net[x])
        degree_diff_gt_net_mean[x] = np.mean(degree_diff_gt_net[x])
    
    print("degree_diff_gt_mean")
    print(degree_diff_gt_mean)
    print("degree_diff_net_mean")
    print(degree_diff_net_mean)
    print("degree_diff_gt_net_mean")
    print(degree_diff_gt_net_mean)

    with open(write_path, mode="wb") as f:
        pickle.dump(doa_values, f)
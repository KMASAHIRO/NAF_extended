import numpy.random
import torch
import os
import pickle
import numpy as np
import random
import h5py
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def listdir(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

def join(*paths):
    return os.path.join(*paths)


class soundsamples(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):
        coor_base = arg_stuff.coor_base
        spec_base = arg_stuff.spec_base
        phase_base = arg_stuff.phase_base
        mean_std_base = arg_stuff.mean_std_base
        phase_std_base = arg_stuff.phase_std_base
        minmax_base = arg_stuff.minmax_base
        num_samples = arg_stuff.pixel_count

        coor_path = os.path.join(coor_base, "points.txt")
        self.max_len = arg_stuff.max_len
        full_path = os.path.join(spec_base, "magnitudes.h5")
        phase_path = os.path.join(phase_base, "phases.h5")

        print("Caching the room coordinate indices, this will take a while....")
        # See https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643
        self.sound_data = []
        self.sound_data = h5py.File(full_path, 'r')
        self.sound_keys = list(self.sound_data.keys())
        self.sound_data.close()
        print("Completed room coordinate index caching")
        self.sound_data = None
        self.phase_data = None

        self.full_path = full_path
        self.phase_path = phase_path

        self.sound_files = []
        self.sound_files_test = []
        self.sound_files_val = []

        train_test_split_path = os.path.join(arg_stuff.split_loc, "complete.pkl")
        with open(train_test_split_path, "rb") as train_test_file_obj:
            train_test_split = pickle.load(train_test_file_obj)
        # use train test split

        self.sound_files = train_test_split[0]
        #self.sound_files_test = train_test_split[1]
        #self.sound_files_val = train_test_split[2]
        self.sound_files_val = train_test_split[1]

        with open(os.path.join(mean_std_base, "magnitude_mean_std.pkl"), "rb") as mean_std_ff:
            mean_std = pickle.load(mean_std_ff)
            print("Loaded mean std")
        self.mean = torch.from_numpy(mean_std[0]).float()[None]
        self.std = 3.0 * torch.from_numpy(mean_std[1]).float()[None]

        # Phase mean is 0 after IF processing
        with open(os.path.join(phase_std_base, "phase_std.pkl"), "rb") as phase_std_ff:
            phase_std = pickle.load(phase_std_ff)
            print("Loaded phase std")
        self.phase_std = 3.0*phase_std

        with open(coor_path, "r") as f:
            lines = f.readlines()
        coords = [x.replace("\n", "").split("\t") for x in lines]
        self.positions = dict()
        for row in coords:
            readout = [float(xyz) for xyz in row[1:]]
            self.positions[row[0]] = [readout[0], readout[1], readout[2]]

        with open(os.path.join(minmax_base, "minmax.pkl"), "rb") as min_max_loader:
            min_maxes = pickle.load(min_max_loader)
            self.min_pos = min_maxes[0][[0, 1]]
            self.max_pos = min_maxes[1][[0, 1]]

        # values = np.array(list(self.positions.values()))
        self.num_samples = num_samples
        self.pos_reg_amt = arg_stuff.reg_eps

    def __len__(self):
        # return number of samples
        return len(self.sound_files)

    def __getitem__(self, idx):
        loaded = False
        while not loaded:
            try:
                if self.sound_data is None:
                    self.sound_data = h5py.File(self.full_path, 'r')

                if self.phase_data is None:
                    self.phase_data = h5py.File(self.phase_path, 'r')

                pos_id = self.sound_files[idx]
                query_str = pos_id

                spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()
                position = (pos_id.split(".")[0]).split("_")
                spec_data = spec_data[:,:,:self.max_len]

                phase_data = torch.from_numpy(self.phase_data[query_str][:]).float()
                phase_data = phase_data[:, :, :self.max_len]

                if random.random()<0.1:
                    # np.log(1e-3) = -6.90775527898213
                    orig_len = spec_data.shape[2]
                    spec_data = torch.nn.functional.pad(spec_data, pad=[0, self.max_len-orig_len, 0, 0, 0, 0], value=-6.90775527898213)
                    phase_data = torch.nn.functional.pad(phase_data, pad=[0, self.max_len-orig_len, 0, 0, 0, 0], value=0.0)

                actual_spec_len = spec_data.shape[2]
                spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
                phase_data = phase_data/self.phase_std

                # 2, freq, time
                sound_size = spec_data.shape
                selected_time = np.random.randint(0, sound_size[2], self.num_samples)
                selected_freq = np.random.randint(0, sound_size[1], self.num_samples)

                non_norm_start = (np.array(self.positions[position[0]])[:2] + np.random.normal(0, 1, 2)*self.pos_reg_amt)
                non_norm_end = (np.array(self.positions[position[1]])[:2]+ np.random.normal(0, 1, 2)*self.pos_reg_amt)
                start_position = (torch.from_numpy((non_norm_start - self.min_pos)/(self.max_pos-self.min_pos))[None] - 0.5) * 2.0
                start_position = torch.clamp(start_position, min=-1.0, max=1.0)
                
                end_position = (torch.from_numpy((non_norm_end - self.min_pos)/(self.max_pos-self.min_pos))[None] - 0.5) * 2.0
                end_position = torch.clamp(end_position, min=-1.0, max=1.0)

                total_position = torch.cat((start_position, end_position), dim=1).float()

                total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()
                
                selected_mag = spec_data[:,selected_freq,selected_time]
                # print(phase_data.shape)
                selected_phase = phase_data[:,selected_freq, selected_time]
                selected_total = torch.cat((selected_mag, selected_phase), dim=0)
                loaded = True

            except Exception as e:
                print(query_str)
                print(e)
                print("Failed to load sound sample")

        return selected_total, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0

    def get_item_teaser(self, reciever_pos, source_pos):
        selected_time = np.arange(0, self.max_len)
        selected_freq = np.arange(0, 256)
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        non_norm_start = np.array(reciever_pos)
        non_norm_end = np.array(source_pos)
        total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

        start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        start_position = torch.clamp(start_position, min=-1.0, max=1.0)
        end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        end_position = torch.clamp(end_position, min=-1.0, max=1.0)
        total_position = torch.cat((start_position, end_position), dim=1).float()

        return total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0

    #def get_item_test(self, idx):
    #    selected_files = self.sound_files_test
    #    if self.sound_data is None:
    #        self.sound_data = h5py.File(self.full_path, 'r')
    #
    #    if self.phase_data is None:
    #        print(self.phase_path)
    #        self.phase_data = h5py.File(self.phase_path, 'r')
    #
    #    pos_id = selected_files[idx]
    #    query_str = pos_id
    #    spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()
    #    phase_data = torch.from_numpy(self.phase_data[query_str][:]).float()
    #
    #    position = (pos_id.split(".")[0]).split("_")
    #
    #    spec_data = spec_data[:, :, :self.max_len]
    #    actual_spec_len = spec_data.shape[2]
    #
    #    spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
    #    phase_data = phase_data / self.phase_std
    #    # 2, freq, time
    #    sound_size = spec_data.shape
    #    self.sound_size = sound_size
    #    self.sound_name = position
    #    selected_time = np.arange(0, sound_size[2])
    #    selected_freq = np.arange(0, sound_size[1])
    #    selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
    #    selected_time = selected_time.reshape(-1)
    #    selected_freq = selected_freq.reshape(-1)
    #
    #    non_norm_start = np.array(self.positions[position[0]])[:2]
    #    non_norm_end = np.array(self.positions[position[1]])[:2]
    #    start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
    #    start_position = torch.clamp(start_position, min=-1.0, max=1.0)
    #    end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
    #    end_position = torch.clamp(end_position, min=-1.0, max=1.0)
    #    total_position = torch.cat((start_position, end_position), dim=1).float()
    #    total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()
    #
    #    # selected_total = spec_data[:, selected_freq, selected_time]
    #    selected_mag = spec_data[:, selected_freq, selected_time]
    #    # print(phase_data.shape)
    #    selected_phase = phase_data[:, selected_freq, selected_time]
    #    selected_total = torch.cat((selected_mag, selected_phase), dim=0)
    #    return selected_total, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0
    
    def get_item_test_train_data(self, idx):
        selected_files = self.sound_files
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')

        if self.phase_data is None:
            print(self.phase_path)
            self.phase_data = h5py.File(self.phase_path, 'r')

        pos_id = selected_files[idx]
        query_str = pos_id
        spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()
        phase_data = torch.from_numpy(self.phase_data[query_str][:]).float()

        position = (pos_id.split(".")[0]).split("_")

        spec_data = spec_data[:, :, :self.max_len]
        actual_spec_len = spec_data.shape[2]

        spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
        phase_data = phase_data / self.phase_std
        # 2, freq, time
        sound_size = spec_data.shape
        self.sound_size = sound_size
        self.sound_name = position
        selected_time = np.arange(0, sound_size[2])
        selected_freq = np.arange(0, sound_size[1])
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        non_norm_start = np.array(self.positions[position[0]])[:2]
        non_norm_end = np.array(self.positions[position[1]])[:2]
        start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        start_position = torch.clamp(start_position, min=-1.0, max=1.0)
        end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        end_position = torch.clamp(end_position, min=-1.0, max=1.0)
        total_position = torch.cat((start_position, end_position), dim=1).float()
        total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

        # selected_total = spec_data[:, selected_freq, selected_time]
        selected_mag = spec_data[:, selected_freq, selected_time]
        # print(phase_data.shape)
        selected_phase = phase_data[:, selected_freq, selected_time]
        selected_total = torch.cat((selected_mag, selected_phase), dim=0)
        return selected_total, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0

    def get_item_val(self, idx):
        selected_files = self.sound_files_val
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')

        if self.phase_data is None:
            print(self.phase_path)
            self.phase_data = h5py.File(self.phase_path, 'r')

        pos_id = selected_files[idx]
        query_str = pos_id
        spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()
        phase_data = torch.from_numpy(self.phase_data[query_str][:]).float()

        position = (pos_id.split(".")[0]).split("_")

        spec_data = spec_data[:, :, :self.max_len]
        actual_spec_len = spec_data.shape[2]

        spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
        phase_data = phase_data / self.phase_std
        # 2, freq, time
        sound_size = spec_data.shape
        self.sound_size = sound_size
        self.sound_name = position
        selected_time = np.arange(0, sound_size[2])
        selected_freq = np.arange(0, sound_size[1])
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        non_norm_start = np.array(self.positions[position[0]])[:2]
        non_norm_end = np.array(self.positions[position[1]])[:2]
        start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        start_position = torch.clamp(start_position, min=-1.0, max=1.0)
        end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        end_position = torch.clamp(end_position, min=-1.0, max=1.0)
        total_position = torch.cat((start_position, end_position), dim=1).float()
        total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

        # selected_total = spec_data[:, selected_freq, selected_time]
        selected_mag = spec_data[:, selected_freq, selected_time]
        # print(phase_data.shape)
        selected_phase = phase_data[:, selected_freq, selected_time]
        selected_total = torch.cat((selected_mag, selected_phase), dim=0)
        return selected_total, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0

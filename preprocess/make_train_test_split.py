import os
import random
import pickle

if __name__ == "__main__":
    path = "./wav_data/raw"

    write_path = "./train_test_split"
    train_test_split = list()

    # Create the directory if it doesn't exist
    if write_path and not os.path.exists(write_path):
        os.makedirs(write_path)
        
    files = os.listdir(path)
    filenames = [f.split(".")[0] for f in files if "wav" in f]
    pos_ids = list()
    for ff in filenames:
        tmp = ff.split("_")
        pos_id = "_".join(tmp[:2])
        if pos_id in pos_ids:
            continue
        else:
            pos_ids.append(pos_id)

    test_len = len(pos_ids)//10
    val_len = len(pos_ids)//10
    pos_ids_shuffle = random.sample(pos_ids, len(pos_ids))
    
    train_test_split.append(pos_ids_shuffle[:-(test_len + val_len)])
    train_test_split.append(pos_ids_shuffle[-(test_len + val_len):-val_len])
    train_test_split.append(pos_ids_shuffle[-val_len:])
    
    with open(os.path.join(write_path, "complete.pkl"), mode="wb") as f:
        pickle.dump(train_test_split, f)
# MultiChannel_RIR_Generation

Source code for "Development of a Microphone Array Room Impulse Response Dataset for Evaluating Multichannel Acoustic Generation (多チャンネル音響生成を評価するためのマイクロフォンアレイ室内インパルス応答データセットの構築)".  
This source code is based on [Learning Neural Acoustic Fields](https://github.com/aluo-x/Learning_Neural_Acoustic_Fields).
  
Neural acoustic fields learn from a Room Impulse Response (RIR) dataset and can estimate RIR at arbitrary positions. We extended Neural Acoustic Fields to be able to learn from multichannel RIRs.  
Additionally, we made it possible to construct datasets through Pyroomacoustics simulations and evaluate multichannel RIR estimation based on the accuracy of Direction of Arrival (DoA), i.e., sound source direction estimation.

## Requirements (in addition to the usual python stack)
- Pytorch 1.9 (1.10 should work as well)
- Pyroomacoustics 0.7.3
- h5py

## Project structure
- `model`: Extended NAF network
- `model_pipeline`: Train, test, and evaluation of the network
- `preprocess`: Preprocessing the RIR dataset for the network
- `simulation`: Pyroomacoustics simulation for creating dataset

## Dataset
### Simulation Data
- Creating a simulation RIR dataset using Pyroomacoustics  
  - ```
    python ./simulation/simulation.py
    ```
  Modifying the simulation environment by adjusting the following parameters within `./simulation/simulation.py`
  https://github.com/KMASAHIRO/MultiChannel_RIR_Generation/blob/31c707729db558375d6227934152a11a77a89a00/simulation/simulation.py#L14-L39

### Real data
- Downloading the real RIR dataset from [Google Drive](https://drive.google.com/file/d/1M0RVo_trCFqszuTthWKwhvr5cqUG-M05/view?usp=sharing)
  - ```
    gdown https://drive.google.com/uc?id=1M0RVo_trCFqszuTthWKwhvr5cqUG-M05  
    unzip real_wav_data.zip
    ```

## Usage
### Preprocess
- Splitting the dataset into training and test dataset  
  - ```
    python ./preprocess/make_train_test_split.py
    ```
- Preprocessing the RIR waveform data to match the format of the network output (e.g., by converting it into spectrograms)
  - ```
    python ./preprocess/make_data.py
    ```
### Train
- Training the network  
  <br>
  For more details on options, refer to `./model_pipeline/options.py`.  
  The trained model will be saved in the directory specified by `save_loc` and `exp_name` options.
  - ```
    python ./model_pipeline/train/train.py --exp_name sim_data_exp --epochs 300 --phase_alpha 3.0 --dir_ch 4
    ```
### Test
- Performing inference on the test data  
  <br>
  The options must be the same as those used in training.  
  The inference output will be saved in the directory specified by the `--save_loc` and `--inference_loc` options.
  - ```
    python ./model_pipeline/test/test.py --exp_name sim_data_exp --epochs 300 --phase_alpha 3.0 --dir_ch 4
    ```
- Perform inference on the train data  
  - ```
    python ./model_pipeline/test/test_train_data.py --exp_name sim_data_exp --epochs 300 --phase_alpha 3.0 --dir_ch 4
    ```
### Evaluation
- Computing the spectral loss from the inference results on the test data  
  <br>
  The options must be the same as those used in training.  
  The results will be printed to the standard output.
  - ```
    python ./model_pipeline/evaluation/compute_spectral_loss.py --exp_name sim_data_exp --epochs 300 --phase_alpha 3.0 --dir_ch 4
    ```
- Computing the T60-error from the inference results on the test data  
  <br>
  The options must be the same as those used in training.  
  The results will be printed to the standard output.
  - ```
    python ./model_pipeline/evaluation/compute_T60_err.py --exp_name sim_data_exp --epochs 300 --phase_alpha 3.0 --dir_ch 4
    ```
- Computing the DoA error from the inference results on the test data  
  <br>
  Do not specify the options.  
  The results will be printed to the standard output.
  - ```
    python ./model_pipeline/evaluation/compute_DoA_err.py
    ```
  Directly modifying `./model_pipeline/evaluation/compute_DoA_err.py` to change the parameters
  https://github.com/KMASAHIRO/MultiChannel_RIR_Generation/blob/31c707729db558375d6227934152a11a77a89a00/model_pipeline/evaluation/compute_DoA_err.py#L8-L19

## Citation
```
@article{加藤 雅大2024,
  title={多チャンネル音響生成を評価するためのマイクロフォンアレイ室内インパルス応答データセットの構築},
  author={加藤 雅大 and 小島 諒介},
  journal={人工知能学会研究会資料 人工知能基本問題研究会},
  volume={128},
  pages={40-45},
  year={2024},
  doi={10.11517/jsaifpai.128.0_40}
}
```

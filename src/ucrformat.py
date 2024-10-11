import os
import math
import random
import numpy as np
import pandas as pd

from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


class GenerateUCR:
    def __init__(
            self, 
            save_path=None, 
            dataset_name=None,
    ):

        self.save_path = save_path
        self.dataset_name = dataset_name

        datasets = load_classification(self.dataset_name)
        if len(datasets) == 2:
            self.signals, self.targets = datasets
        else:
            self.signals, self.targets, _ = datasets
        self.signals = np.array(self.signals)
        self.signals = self.signals.transpose(0, 2, 1) # (num, timestep, feature)
        self.num_samples = self.signals.shape[0]


    def format_to_npy(self):
        num_instances = range(self.num_samples)
        data_list = []

        for idx in num_instances:
            sample = self._generate_sample(idx)
            data_list.append(sample)
        
        random.seed(42)
        random.shuffle(data_list)

        df_data = pd.DataFrame(data_list)
        np_signals = np.stack(df_data['signal'].values)
        np_targets = np.stack(df_data['target'].values)
        label_encoder = LabelEncoder()
        np_targets = label_encoder.fit_transform(np_targets)

        # save files
        np.save(os.path.join(self.save_path, 'signals.npy'), np_signals)
        np.save(os.path.join(self.save_path, 'targets.npy'), np_targets)
        print('Already Done!')


    def _generate_sample(self, index):
        dict_all = {}
        dict_all["noun_id"] = f"sample_{index}"
        signal = self.signals[index]
        dict_all["signal"] = signal.astype(np.float32)
        dict_all["target"] = self.targets[index].astype(str)
        dict_all["signal_length"] = signal.shape[0]
        dict_all["signal_names"] = np.array([f"feature_{str(i)}" for i in range(signal.shape[1])]).astype(np.string_)
        return dict_all


if __name__ == "__main__":
    # # UCR datasets
    # dataset_name = [
    #     "Blink",
    #     "CBF",
    #     "ElectricDevices",
    #     "EMOPain",
    #     "FordA",
    #     "FreezerRegularTrain",
    #     "GunPointAgeSpan",
    #     "ItalyPowerDemand",
    #     "MoteStrain",
    #     "MotionSenseHAR",
    #     "PenDigits",
    #     "SonyAIBORobotSurface1",
    #     "SonyAIBORobotSurface2",
    #     "SpokenArabicDigits",
    #     "StarLightCurves",
    #     "Strawberry",
    #     "TwoLeadECG",
    #     "TwoPatterns",
    #     "Wafer",
    # ]
    dataset_name = "Blink"
    proj_path = os.getcwd()
    save_path = os.path.join(proj_path, 'data', dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    dict_pre = GenerateUCR(save_path=save_path, dataset_name=dataset_name)
    dict_pre.format_to_npy()

import json
import numpy as np
import os
import pickle
from datetime import datetime
from scipy import stats


class XGBLogger():

    def __init__(self, result_dir: str, description: str):

        self.result_dir = result_dir
        self.description = description

    def _make_resultdir_n_subdirs(self):
        """
        Creates result_dir and datetime named folder inside
        """
        now = datetime.now()
        datename = "-".join([str(now.date())[5:], str(now.hour)])
        self.result_dir = os.path.join(self.result_dir, datename)
        print(self.result_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(os.path.join(self.result_dir, 'preds'))
            os.makedirs(os.path.join(self.result_dir, 'models'))
            os.makedirs(os.path.join(self.result_dir, 'params'))

    def calc_metric(self, metric, preds, labels):
        score = metric(labels, preds)
        with open(os.path.join(self.result_dir, 'meta.txt'), 'a') as meta_file:
            meta_file.writelines(f'val metric score --> {score}' + '\n')

    def save_description(self):
        with open(os.path.join(self.result_dir, 'meta.txt'), 'w') as meta_file:
            meta_file.writelines(self.description + '\n')

    def save_kfold_information(self, kfold):
        with open(os.path.join(self.result_dir, 'meta.txt'), 'a') as meta_file:
            meta_file.write(str(kfold.__dict__))

    def save_model(self, model, i_fold: str):
        with open(os.path.join(self.result_dir, 'models', f'model{i_fold}.pickle'), 'wb') as f:
            pickle.dump(model, f)

    def save_preds(self, train_preds: np.ndarray, val_preds: np.ndarray):
        np.savetxt(os.path.join(self.result_dir, 'preds', 'train_preds.csv'), train_preds, delimiter=",")
        if val_preds is not None:
            np.savetxt(os.path.join(self.result_dir, 'preds', 'val_preds.csv'), val_preds, delimiter=",")

    def save_params(self, booster_params: dict, train_params: dict, log_params: dict):
        with open(os.path.join(self.result_dir, 'params', 'booster_params.csv'), 'w') as f:
            json.dump(booster_params, f)
        with open(os.path.join(self.result_dir, 'params', 'train_params.csv'), 'w') as f:
            json.dump(train_params, f)
        with open(os.path.join(self.result_dir, 'params', 'log_params.csv'), 'w') as f:
            json.dump(log_params, f)
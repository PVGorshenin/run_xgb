import sys
sys.path.append('..')

import numpy as np
import os
from logger import XGBLogger
from pytest import fixture
from scipy import stats
from sklearn.metrics import accuracy_score, mean_absolute_error


@fixture
def get_preds():
    return np.array([[0, 0, 0],
                     [1, 1, 1],
                     [1, 1, 1]])


@fixture
def get_labels():
    return np.array([[0], [1], [0]])


def clf_accuracy(labels, preds):
    return accuracy_score(labels, stats.mode(preds, 0)[0].reshape((-1, 1)))


def test_clf_metric(get_preds, get_labels, tmpdir):
    res_path = tmpdir.mkdir("res_path")
    logger = XGBLogger(res_path, 'descr')
    logger.calc_metric(clf_accuracy, get_preds, get_labels)
    assert len(os.listdir(res_path))


def test_reg_metric(get_preds, get_labels, tmpdir):
    res_path = tmpdir.mkdir("res_path")
    logger = XGBLogger(res_path, 'descr')
    logger.calc_metric(mean_absolute_error, get_preds.mean(0), get_labels)
    assert len(os.listdir(res_path))
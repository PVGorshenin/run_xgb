import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from check_importance import _measure_target_response, _except_zero_div
from pytest import fixture

@fixture
def get_start_preds_zeros():
    return pd.Series(np.array([0, 1]))


@fixture
def get_start_preds_nan():
    return pd.Series(np.array([np.NaN, 1]))

@fixture
def get_changed_preds():
    return pd.Series(np.array([1e-6, 1.1]))


def test_zero_preds_rel(get_start_preds_zeros, get_changed_preds):
    start_preds = _except_zero_div(get_start_preds_zeros)
    rel_delta = _measure_target_response(start_preds, get_changed_preds,
                                         target_measure_method='rel', agg_method='mean')
    assert np.isclose(rel_delta, .05)


def test_niull_preds_rel(get_start_preds_nan, get_changed_preds):
    #NaN не учитываются в усреднении
    rel_delta = _measure_target_response(get_start_preds_nan, get_changed_preds,
                                         target_measure_method='rel', agg_method='mean')
    assert np.isclose(rel_delta, .1)



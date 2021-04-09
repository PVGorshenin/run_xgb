import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse.csr import csr_matrix
from typing import Tuple, Union
from logging_utils import common_logging


def _get_init_data(train, val, train_cols,  kfold, booster_params):
    preds_train = np.zeros(train.shape[0])
    if val is not None:
        if isinstance(val, pd.DataFrame):
            dval = xgb.DMatrix(val[train_cols])
        if isinstance(val, csr_matrix):
            dval = xgb.DMatrix(val)
        if booster_params['objective'] == 'multi:softprob':
            preds_val = np.zeros([kfold.n_splits, val.shape[0], booster_params['num_class']])
            preds_train = np.zeros([train.shape[0], booster_params['num_class']])
            return (preds_train, dval, preds_val)
        preds_val = np.zeros([kfold.n_splits, val.shape[0]])
        return (preds_train, dval, preds_val)
    return (preds_train, None, None)


def _get_kfold_data(train, train_index, test_index, train_cols, labels):
    if isinstance(train, pd.DataFrame):
        dtrain_train = xgb.DMatrix(train[train_cols].iloc[train_index, :], labels.iloc[train_index].values)
        dtrain_val = xgb.DMatrix(train[train_cols].iloc[test_index, :], labels.iloc[test_index].values)
    if isinstance(train, csr_matrix):
        dtrain_train = xgb.DMatrix(train[train_index, :], labels.iloc[train_index])
        dtrain_val = xgb.DMatrix(train[test_index, :], labels.iloc[test_index])
    return dtrain_train, dtrain_val


def _predict_n_save_val(preds_val: np.ndarray, xgb_model, dval, logger, booster_params, kfold, i_fold):
    '''
    Сохраняем по эпохам для мультикласса, иначе целиком

    :param preds_val:  n_splits * n_examles * n_targets для мультикласса с выходом n_examles * n_targets
                        n_splits * n_examles в других случаях
    '''
    preds_val[i_fold] = xgb_model.predict(dval)
    if booster_params['objective'] == 'multi:softprob':
        logger.save_val_preds(preds_val[i_fold], i_fold)
    elif i_fold == kfold.n_splits-1:
        logger.save_val_preds(preds_val, 0)
    return preds_val


@common_logging
def run_xgb(train: Union[pd.DataFrame, csr_matrix], val: Union[pd.DataFrame, csr_matrix], train_cols: list,
            train_labels: Union[pd.DataFrame, pd.Series], val_labels: Union[pd.DataFrame, pd.Series],
            booster_params: dict, train_params: dict, log_params: dict,
            kfold, metric, logger) -> Tuple[np.ndarray]:
    """
    Runs xgboost in KFold cycle

    Simple runner. One train, one test.
    :param train: dataset to pass in kfold
    :param val: dataset to predict (no usage in training)
    :param logger: always None, overwritten in common_logging function
    :return: preds_train, preds_val
    """
    preds_train, dval, preds_val = _get_init_data(train, val, train_cols, kfold, booster_params)
    for i_fold, (train_index, test_index) in enumerate(kfold.split(train)):
        dtrain_train, dtrain_val = _get_kfold_data(train, train_index, test_index, train_cols, train_labels)
        evallist = [(dtrain_train, 'train'), (dtrain_val, 'val')]
        xgb_model = xgb.train(booster_params,
                              dtrain_train,
                              num_boost_round=train_params['num_boost_round'],
                              evals=evallist,
                              verbose_eval=train_params['verbose_eval'],
                              early_stopping_rounds=train_params['early_stopping_rounds'])
        logger.save_model(xgb_model, i_fold)
        preds_train[test_index] = xgb_model.predict(dtrain_val)
        if val is not None:
            preds_val = _predict_n_save_val(preds_val, xgb_model, dval, logger, booster_params, kfold, i_fold)
            if val_labels is not None:
                logger.calc_metric(metric, preds_val[i_fold, :, :], val_labels.values)
    return (preds_train, preds_val)
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple, Union
from logger import XGBLogger


def _get_val_data(val_df, kfold):
    if val_df is not None:
        dval = xgb.DMatrix(val_df)
        preds_val = np.zeros([kfold.n_splits, val_df.shape[0]])
        return (dval, preds_val)
    return (None, None)


def run_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, train_cols: list, label_cols: Union[str, list],
            booster_params: dict, train_params: dict, log_params: dict, kfold, metric) -> Tuple[np.ndarray]:
    """
    Runs xgboost in KFold cycle

    Simple runner. One train, one test.
    """
    logger = XGBLogger(result_dir=log_params['result_path'],
                       description=log_params['description'])
    logger._make_resultdir_n_subdirs()
    logger.save_description()
    preds_train = np.zeros(train_df.shape[0])
    dval, preds_val = _get_val_data(val_df[train_cols], kfold)
    for i_fold, (train_index, test_index) in enumerate(kfold.split(train_df)):
        dtrain_train = xgb.DMatrix(train_df[train_cols].iloc[train_index, :], train_df[label_cols].iloc[train_index])
        dtrain_val = xgb.DMatrix(train_df[train_cols].iloc[test_index, :], train_df[label_cols].iloc[test_index])

        evallist = [(dtrain_train, 'train'), (dtrain_val, 'val')]
        xgb_model = xgb.train(booster_params,
                              dtrain_train,
                              num_boost_round=train_params['num_boost_round'],
                              evals=evallist,
                              verbose_eval=train_params['verbose_eval'],
                              early_stopping_rounds=train_params['early_stopping_rounds'])
        logger.save_model(xgb_model, i_fold)
        preds_train[test_index] = xgb_model.predict(dtrain_val)
        if val_df is not None: preds_val[i_fold, :] = xgb_model.predict(dval)
    logger.save_preds(preds_train, preds_val)
    logger.save_params(booster_params, train_params, log_params)
    if (val_df is not None) and (label_cols in val_df.columns):
        logger.calc_metric(metric, preds_val, val_df[label_cols].values)
    return (preds_train, preds_val)
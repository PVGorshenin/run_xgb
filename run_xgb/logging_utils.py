from .logger import XGBLogger
from functools import wraps



def _save_val_preds(logger, preds_val, booster_params, kfold, i_fold):
    '''
    Сохраняем по эпохам для мультикласса, иначе целиком

    :param preds_val:  n_splits * n_examles * n_targets для мультикласса с выходом n_examles * n_targets
                        n_splits * n_examles в других случаях
    '''

    if booster_params['objective'] == 'multi:softprob':
        logger.save_val_preds(preds_val[i_fold], i_fold)
    elif i_fold == kfold.n_splits-1:
        logger.save_val_preds(preds_val, 0)


def common_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = XGBLogger(result_dir=kwargs['log_params']['result_path'],
                           description=kwargs['log_params']['description'])
        logger._make_resultdir_n_subdirs()
        logger.save_description()

        preds_train, preds_val, model_lst = func(*args, **kwargs)

        for i_fold in range(kwargs['kfold'].n_splits):
            _save_val_preds(logger, preds_val, kwargs['booster_params'], kwargs['kfold'], i_fold)
            logger.save_model(model_lst[i_fold], i_fold)
            if (kwargs['val'] is not None) & (kwargs['val_labels'] is not None):
                logger.calc_metric(kwargs['metric'], preds_val[i_fold, :], kwargs['val_labels'].values)

        logger.save_train_preds(preds_train)
        logger.save_params(kwargs['booster_params'], kwargs['train_params'], kwargs['log_params'])
        return preds_train, preds_val
    return wrapper
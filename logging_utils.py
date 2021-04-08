from .logger import XGBLogger
from functools import wraps


def common_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = XGBLogger(result_dir=kwargs['log_params']['result_path'],
                           description=kwargs['log_params']['description'])
        logger._make_resultdir_n_subdirs()
        logger.save_description()
        kwargs['logger'] = logger

        preds_train, preds_val = func(*args, **kwargs)

        logger.save_train_preds(preds_train)
        logger.save_params(kwargs['booster_params'], kwargs['train_params'], kwargs['log_params'])
        return preds_train, preds_val
    return wrapper
from dateutil.relativedelta import relativedelta
from functools import wraps
from .time_logger import TimeXGBLogger



def eat_by_month_generator(df, n_months_to_eat, first_step_val_date, printing=True):
    '''

    :param df: входной датафрейм
    :param n_months_to_eat: сколько месяцев пожирать
    :param first_step_val_date: дата в формате Y-m-d, обозначающ. начало месяца, след. за минимальным
    :return:
    '''
    for i_month in range(n_months_to_eat):
        border_date = first_step_val_date + relativedelta(months=i_month)
        after_date = df['fecha_dato'] >= border_date
        if printing:
            print('Val-part starts with -->', border_date.date())
            # print(f'Train part size--> {~after_date.sum()}    Val part size --> {after_date.sum()}\n')
        yield (after_date, border_date)



def time_validation_n_logger(func, df, n_months_to_eat: int, first_step_val_date, filter_users: bool, user_col=None):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = TimeXGBLogger(result_dir=kwargs['log_params']['result_path'],
                               description=kwargs['log_params']['description'])
        logger._make_resultdir_n_subdirs()
        basic_dir = logger.result_dir
        logger.save_description()

        time_eater = eat_by_month_generator(df, n_months_to_eat, first_step_val_date)
        logger.save_time_eater_params(n_months_to_eat, first_step_val_date)
        logger.save_params(kwargs['booster_params'], kwargs['train_params'], kwargs['log_params'])

        for (after_border, border_date) in time_eater:
            logger._make_timestep_dir(border_date)
            kwargs['logger'] = logger
            kwargs['train_df'] = df[~after_border]
            val_sel = after_border
            if filter_users:
                train_users = df.loc[~after_border, user_col].unique()
                val_sel &= (~df[user_col].isin(train_users))
            kwargs['val_df'] = df[val_sel]
            preds_train, preds_val = func(*args, **kwargs)

            logger.save_train_preds(preds_train)
            logger.result_dir = basic_dir
    return wrapper



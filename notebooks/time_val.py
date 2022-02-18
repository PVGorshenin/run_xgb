import numpy as np
from dateutil.relativedelta import relativedelta


def eat_by_days_generator(df, n_days_in_val, n_days_in_train_val, first_step_val_date, date_col, printing=True):

    n_steps = int(np.ceil((df[date_col] >= first_step_val_date).sum() / n_days_in_val))

    for i_step in range(n_steps):
        val_border_date = first_step_val_date + relativedelta(days=i_step * n_days_in_val)
        val_right_border = val_border_date + relativedelta(days=n_days_in_val)

        train_val_border_date = val_border_date - relativedelta(days=n_days_in_train_val)

        val_mask = (df[date_col] >= val_border_date) & (df[date_col]<=val_right_border)
        train_val_mask = (df[date_col] >= train_val_border_date) & (df[date_col] < val_border_date)

        if printing:
            print('Val-part starts with -->', val_border_date.date())
        yield (val_mask, train_val_mask)


def eat_by_points_generator(df, n_points_in_val, n_points_in_train_val, starting_point, date_col, printing=True):
    n_steps = int(np.ceil((df.shape[0] - starting_point) / n_points_in_val)) - 1
    curr_point = starting_point
    for i_step in range(n_steps):
        val_border_date = df['date'].iloc[curr_point]
        jump = min(n_points_in_val, df.shape[0] - curr_point - 1)

        val_right_border = df['date'].iloc[curr_point + jump]

        train_val_border_date = df['date'].iloc[curr_point - n_points_in_train_val]

        val_mask = (df[date_col] >= val_border_date) & (df[date_col] < val_right_border)
        train_val_mask = (df[date_col] >= train_val_border_date) & (df[date_col] < val_border_date)

        if printing:
            print('Val-part starts with -->', val_border_date.date())

        curr_point += n_points_in_val

        yield (val_mask, train_val_mask)



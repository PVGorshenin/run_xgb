import pandas as pd
import xgboost as xgb


def _increase_col_value(df, input_col, increase_koeff, input_change_method):
    if input_change_method=='rel':
        percent = .01
        df[input_col] += increase_koeff * percent * df[input_col]
        return df
    if input_change_method=='abs':
        df[input_col] += increase_koeff
    return df


def _measure_target_response(start_preds: pd.Series, changed_preds:pd.Series, target_measure_method, agg_method):
    assert target_measure_method in ['rel', 'abs', 'mass_rel']

    if target_measure_method=='rel':
        rel_delta = (changed_preds - start_preds) / start_preds
        return rel_delta.agg(agg_method)
    if target_measure_method=='abs':
        abs_delta = changed_preds - start_preds
        return abs_delta.agg(agg_method)
    if target_measure_method == 'mass_rel':
        mass_rel_delta = ((changed_preds - start_preds).sum()) / start_preds.sum()
        return mass_rel_delta


def _except_zero_div(start_preds):
    start_preds[start_preds == 0] += 1e-6
    return start_preds


def check_importance(df, model, input_col: str, input_change_method: str,  target_measure_method: str,
                      increase_koeff: int, agg_method:str):
    """
    Оценка важности признака

    Проверяет опытным путём количественный отклик таргета на количественное
    изменнение входного параметра
    :param input_change_method: {rel, abs}
    :param target_measure_method: {rel, abs, mass_rel}
    :return:
    """
    start_preds = pd.Series(model.predict(xgb.DMatrix(df[model.feature_names])))
    start_preds = _except_zero_div(start_preds)
    df = _increase_col_value(df, input_col, increase_koeff, input_change_method)
    changed_preds = pd.Series(model.predict(xgb.DMatrix(df[model.feature_names])))
    print('Start preds {} --> {:.2f}    changed preds {} --> {:.2f}'.format(
        agg_method, start_preds.agg(agg_method), agg_method, changed_preds.agg(agg_method)))
    target_changes =  _measure_target_response(start_preds, changed_preds, target_measure_method, agg_method)
    print('{:.4f}'.format(target_changes))
    return target_changes

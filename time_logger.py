import os
from datetime import datetime
from .logger import XGBLogger


class TimeXGBLogger(XGBLogger):

    def _make_resultdir_n_subdirs(self):
        """
        Creates result_dir and datetime named folder inside
        """
        now = datetime.now()
        datename = "-".join([str(now.date())[5:], str(now.hour)])
        self.result_dir = os.path.join(self.result_dir, datename,)
        print(self.result_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(os.path.join(self.result_dir, 'params'))

    def _make_timestep_dir(self, border_date):
        self.result_dir = os.path.join(self.result_dir, '-'.join([str(border_date.year), str(border_date.month)]))
        if not os.path.isdir(self.result_dir):
            os.makedirs(os.path.join(self.result_dir, 'preds'))
            os.makedirs(os.path.join(self.result_dir, 'models'))

    def save_time_eater_params(self, n_months_to_eat, first_step_val_date):
        with open(os.path.join(self.result_dir, 'params', 'eater_params.txt'), 'w') as f:
            f.writelines([f'n_months_to_eat --> {n_months_to_eat}',
                          f'first_step_val_date{first_step_val_date}'])

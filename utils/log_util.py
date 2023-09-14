from datetime import datetime
import os
from utils import mk_dir
from tensorboardX import SummaryWriter
from tabulate import tabulate


class Log:
    def __init__(self, args):
        log_root = mk_dir(f'./log/{args.mode}')
        today = str(datetime.today().strftime('%y%m%d_%H-%M'))
        self.log_dir = mk_dir(os.path.join(log_root, f'log_time_series_{today}'))
        self.log_file = os.path.join(self.log_dir, 'log.txt')
        self.model_save_dir = mk_dir(os.path.join(self.log_dir, 'model_save')) if args.save_model else None
        self.img_save_dir = mk_dir(os.path.join(self.log_dir, 'image_results')) if args.save_image else None
        self.writer = SummaryWriter(os.path.join(self.log_dir, args.description))

    def logging(self, string):
        print(string)
        with open(self.log_file, 'at', encoding='utf-8') as f:
            f.write(str(string))
            f.write('\n')

    def print_table(self, header, table):
        self.logging('\n')
        header = list(map(str, header))
        table = [list(map(str, x)) for x in table]
        self.logging(tabulate(table, header, tablefmt='github', stralign='center', numalign='center'))

    def print_label_rmse_table(self, reg_rmse_dict):
        k_sort = sorted(reg_rmse_dict.keys())
        header = ['label'] + k_sort
        table = [['rmse']+[f'{reg_rmse_dict[k]:.3f}' for k in k_sort]]
        self.print_table(header, table)


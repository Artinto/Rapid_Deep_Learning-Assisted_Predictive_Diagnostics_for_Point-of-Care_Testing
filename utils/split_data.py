import os
import glob
import argparse
import time as t
import random as rd
from shutil import copytree

from utils.util import mk_dir

parser = argparse.ArgumentParser(description='Preprocess Diagnostic-Kit Image')

parser.add_argument('--root', type=str, default='../workdir/all_dataset/LJH_prof/latest')
parser.add_argument('--input_dirname', type=str, default='all')
parser.add_argument('--train_dirname', type=str, default='train')
parser.add_argument('--eval_dirname', type=str, default='eval')

args = parser.parse_args()


def check_data(dirs: list, density: int, split_info: tuple):
    if len(dirs) != sum(split_info):
        raise Exception("density: {} -> all: {} | but current setting: <train : eval = {} : {}>".format(
            density, len(dirs), *split_info))


def split_data(dirs: list, density: int, split_info: tuple, args):
    rd.shuffle(dirs)
    train = dirs[:split_info[0]]
    eval = dirs[split_info[0]:]
    for d in train:
        copytree(os.path.join(args.input_dirname, str(density), d),
                 os.path.join(args.train_dirname, str(density), d))
    for d in eval:
        copytree(os.path.join(args.input_dirname, str(density), d),
                 os.path.join(args.eval_dirname, str(density), d))


if __name__ == '__main__':
    start = t.time()

    rd.seed(777)

    args.input_dirname = os.path.join(args.root, args.input_dirname)       # input
    args.train_dirname = os.path.join(args.root, args.train_dirname)     # output 1
    args.eval_dirname = os.path.join(args.root, args.eval_dirname)       # output 2
    mk_dir(args.train_dirname)
    mk_dir(args.eval_dirname)

    density_dict = {
        0: (160,40),
        10000: (80,20),
        25000: (21,7),
        50000: (80,20),
        75000: (21,7),
        100000: (80,20),
        150000: (16,4),
        200000: (80,20),
        250000: (4,1),
    }
    print(density_dict, end='\n\n')
    for i, (path, dirs, files) in enumerate(os.walk(args.input_dirname)):
        if i == 0 or dirs == [] or files != []: continue
        density = int(os.path.split(path)[1])
        print('* density :', density)
        split_info = density_dict[density]              # (num_train, num_eval) = split_info

        check_data(dirs, density, density_dict[density])
        split_data(dirs, density, split_info, args)

    print('\n - Time: {:.0f} sec'.format((t.time() - start)))


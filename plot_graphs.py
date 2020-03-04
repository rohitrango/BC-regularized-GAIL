import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse
import torch
import os
import sys
from baselines.common import plot_util as pu
from baselines.common.plot_util import COLORS

import matplotlib as mplot
mplot.rcParams.update({'font.size': 16})



parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', required=True,
            help='Get filenames')
parser.add_argument('--legend', nargs='+', default=[],
            help='Legend values')
parser.add_argument('--max_steps', default=0, type=int)
args = parser.parse_args()

def check_last_name(result):
    path = result.dirname
    splits = path.split('/')
    for sp in splits[::-1]:
        if sp == '':
            continue
        try:
            sp = int(sp)
        except:
            return sp
    return ''


def main():
    """
    Plot the plots inside the folder given
    """
    # Now plot the common things
    splits = args.files[0].split('/')
    if splits[-1] == '':
        splits = splits[-2]
    else:
        splits = splits[-1]
    env = splits
    results = pu.load_results(args.files, )
    fig = pu.plot_results(results, average_group=True,
            shaded_err=False,
            shaded_std=True,
            group_fn=lambda _: check_last_name(_),
            split_fn=lambda _: '', figsize=(10, 10))

    # Add results for behaviour cloning if present
    allfiles = []
    for file in args.files:
        for r, dirs, files in os.walk(file):
            txtfiles = list(filter(lambda x: x.endswith('bc.txt'), files))
            allfiles.extend(list(map(lambda x: os.path.join(r, x), txtfiles)))

    if allfiles != []:
        bcreward = []
        for file in allfiles:
            with open(file, 'r') as fi:
                meanrew = float(fi.readlines()[0])
                bcreward.append(meanrew)

        # Get mean and std
        mean = np.mean(bcreward)
        std = np.std(bcreward)
        idxcolor=4
        plt.plot([0, args.max_steps], [mean, mean], label='BC', color=COLORS[idxcolor])
        plt.fill_between([0, args.max_steps], [mean - std, mean - std], [mean + std, mean + std], alpha=0.2, color=COLORS[idxcolor])

    plt.xlabel('Number of steps')
    plt.ylabel('Reward')
    plt.title(env)
    if args.legend != []:
        if allfiles != []:
            args.legend.append('BC')
        plt.legend(args.legend)
    plt.ticklabel_format(useOffset=1)
    plt.savefig('{}.png'.format(env), bbox_inches='tight', )
    print("saved ", env)


if __name__ == "__main__":
    main()

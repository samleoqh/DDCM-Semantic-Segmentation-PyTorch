from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import datetime


class Config(object):
    """Base training configuration class.
    """
    model = 'DDCM_R50'
    suffix_note = ''

    # data set parameters
    dataset = 'Potsdam'
    bands = ['RGB']  # ['IRRG', 'DSM']  # mulit-bands
    k_folder = 5
    k = 0
    input_size = [256, 256]
    train_samples = 5000
    val_samples = 1000
    train_batch = 5
    val_batch = 5
    pre_norm = False

    # training hyper parameters
    lr = 8.5e-5 / np.sqrt(2)
    lr_decay = 0.9
    weight_decay = 2e-5
    momentum = 0.95
    max_iter = 1e8

    # check point parameters
    ckpt_path = '/media/liu/diskb/ckpt'
    snapshot = ''
    print_freq = 100
    save_pred = True
    save_rate = 0.1

    def __init__(self, net_name=model, data=dataset, bands_list=bands, kf=1, note=''):
        self.model = net_name
        self.dataset = data
        self.bands = bands_list
        self.k = kf
        self.suffix_note = note

        check_mkdir(self.ckpt_path)
        check_mkdir(os.path.join(self.ckpt_path, self.model))

        bandstr = '-'.join(self.bands)
        if self.k_folder is not None:
            subfolder = self.dataset + '_' + bandstr + '_kf-' + str(self.k_folder) + '-' + str(self.k)
        else:
            subfolder = self.dataset + '_' + bandstr
        if note != '':
            subfolder += '-'
            subfolder += note

        check_mkdir(os.path.join(self.ckpt_path, self.model, subfolder))
        self.save_path = os.path.join(self.ckpt_path, self.model, subfolder)

    def display(self):
        """printout all configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def write2txt(self):
        file = open(os.path.join(self.save_path,
                                 str(datetime.datetime.now()) + '.txt'), 'w')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                line = "{:30} {}".format(a, getattr(self, a))
                file.write(line + '\n')


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

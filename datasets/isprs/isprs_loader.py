from datasets.data_augmt import *

import os
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

"""
ISPRS datasets folder structure as below:
---
  |-----Potsdam_DIR
            |------2_Ortho_RGB   
            |------2_Ortho_IRRG 
            |------5_Labels_for_participants
            |------5_Labels_for_participants_no_Boundary

  |-----Vaihingen_DIR
            |------top
            |------gts_for_participants
            |------gts_eroded_for_participants

"""

# all classes
LABELS = ["Buildings", "Trees", "Low veg.", "Clutter", "Road surf", "Cars"]

# Path to data dir
DATA_FOLDER = {
    'Potsdam': {
        'ROOT': '/media/liu/diskb/data/Potsdam',
        'RGB': '2_Ortho_RGB/top_potsdam_{}_RGB.tif',
        'IRRG': '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif',
        'DSM': '1_DSM_normalisation/dsm_potsdam_0{}_normalized_lastools.jpg',
        'GT': '5_Labels_all/top_potsdam_{}_label.tif',
        'GT_nb': '5_Labels_all_noBoundary/top_potsdam_{}_label_noBoundary.tif',
        'IDs': [
            '2_10',
            '2_11', '2_12',
            '3_10', '3_11', '3_12',
            '4_10', '4_11', '4_12',
            '5_10', '5_11', '5_12',
            '6_10', '6_11', '6_12',
            '7_10', '7_12', '7_11',
            '6_7', '6_8', '6_9',
            '7_7', '7_8', '7_9',
        ],

        'IDs_v2': [
            '5_11', '6_9', '7_11'
        ],  # for local test set

        'IDs_fv': [
            '4_10', '7_10'
        ], # for local validation fixed

        'IDs_test': [
            '2_13', '2_14',
            '3_13', '3_14',
            '4_13', '4_14', '4_15',
            '5_13', '5_14', '5_15',
            '6_13', '6_14', '6_15',
            '7_13'
        ],  # hold-out test set
    },

    'Vaihingen': {
        'ROOT': '/media/liu/diskb/data/Vaihingen/ISPRS_semantic_labeling_Vaihingen',
        'IRRG': 'top/top_mosaic_09cm_area{}.tif',
        'DSM': 'dsm/dsm_09cm_matching_area{}.tif',
        'GT': 'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/top_mosaic_09cm_area{}.tif',
        'GT_nb': 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE/top_mosaic_09cm_area{}_noBoundary.tif',
        'IDs': [
            '1', '3',
            '5', '7',
            '11', '13', '15', '17',
            '21', '23', '26', '28',
            '30', '32', '34', '37'
        ],

        'IDs_v2': [
            '30',
            '5', '15', '21'
        ],  # for local test set, not the validation set

        'IDs_fv': [
            '7', '28'
        ], # for local validation fixed

        'IDs_test': [
            '2', '4', '6', '8',
            '10', '12', '14', '16', '20',
            '22', '24', '27', '29', '31',
            '33', '35', '38'
        ],  # hold-out test set
    },
}

# original color map
palette_org = {
    0: (0, 0, 255),  # Buildings (blue)
    1: (0, 255, 0),  # Trees (green)
    2: (0, 255, 255),  # Low vegetation (cyan)
    3: (255, 0, 0),  # Clutter (red)
    4: (255, 255, 255),  # Impervious surfaces (white)
    5: (255, 255, 0),  # Cars (yellow)
    6: (0, 0, 0)  # Undefined (black)
}

# customised palette for visualization, easier for reading in paper
palette_vsl = {
    0: (0, 0, 255),  # Buildings (blue)
    1: (0, 255, 0),  # Trees (green)
    2: (0, 200, 200),  # Low vegetation (dark cyan)
    3: (255, 0, 0),  # Clutter (red)
    4: (255, 255, 255),  # Impervious surfaces (white)
    5: (128, 128, 0),  # Cars (dark yellow)
    6: (0, 0, 0)  # Undefined (black)
}

IMG = 'image_file'  # RGB or IRRG
# DSM = 'dsm_file'  # DSM band
GT = 'ground_truth'
GTNB = 'GT_noboundary'


def get_train_test_files(data_folder=DATA_FOLDER, name='Potsdam', bands='RGB'):
    test_ids = data_folder[name]['IDs_test']
    train_ids = [item for item in data_folder[name]['IDs'] if item not in test_ids]

    img_ids = os.path.join(data_folder[name]['ROOT'], data_folder[name][bands])
    gt_ids = os.path.join(data_folder[name]['ROOT'], data_folder[name]['GT'])
    gt_nb_ids = os.path.join(data_folder[name]['ROOT'], data_folder[name]['GT_nb'])

    train_files = {
        IMG: [img_ids.format(id) for id in train_ids],
        GT: [gt_ids.format(id) for id in train_ids],
        GTNB: [gt_nb_ids.format(id) for id in train_ids]
    }

    test_files = {
        IMG: [img_ids.format(id) for id in test_ids],
        GT: [gt_ids.format(id) for id in test_ids],
        GTNB: [gt_nb_ids.format(id) for id in test_ids]
    }

    return train_files, test_files


def sanity_check():
    for name in ['Potsdam', 'Vaihingen']:
        print('checking', name)
        for band in ['RGB', 'IRRG']:
            if name == 'Vaihingen' and band == 'RGB':
                print('skip RGB')
                continue
            print('check bands:', band)
            train_files, test_files = get_train_test_files(name=name, bands=band)
            print('total train files:{} and test files:{}'.format(len(train_files[IMG]),
                                                                  len(test_files[IMG])))

            for f in train_files[IMG] + train_files[GT] + train_files[GTNB] + \
                     test_files[IMG] + test_files[GT] + test_files[GTNB]:
                if not os.path.isfile(f):
                    raise KeyError('{} not found or not a file !'.format(f))
            print('-------pass---------')


# k-folder or random split train, val, and test set with mixed bands
def split_train_val_test_sets(data_folder=DATA_FOLDER, name='Potsdam', bands=['RGB'], KF=None, k=1, seeds=69278):
    sub_ids = data_folder[name]['IDs_v2']  # this is for local hold-out test
    train_ids = [item for item in data_folder[name]['IDs']]
    test_ids = data_folder[name]['IDs_test']  # which is predefined, in stead of randomly selected

    train_ids = [item for item in train_ids if
                 item not in test_ids + sub_ids]  # remove local test and hold-out test ids from list

    # which will be further splitted into train and val
    if KF is None:
        train_id, val_id = train_test_split(train_ids, test_size=0.1, random_state=seeds)
    else:
        kf = KFold(n_splits=KF, shuffle=True, random_state=seeds)
        train_ids = np.array(train_ids)
        idx = list(kf.split(np.array(train_ids)))
        if k >= KF:  # k should not be out of KF range, otherwise set k = 0
            k = 0
        train_id, val_id = list(train_ids[idx[k][0]]), list(train_ids[idx[k][1]])

    if name == 'Vaihingen':
        if 'DSM' in bands:
            train_id.remove('11')  # area11 dsm file not valid, remove it from training id list

    print('train ids: ', train_id)
    print('val_id ids: ', val_id)

    img_ids = [os.path.join(data_folder[name]['ROOT'], data_folder[name][band]) for band in bands]
    gt_ids = os.path.join(data_folder[name]['ROOT'], data_folder[name]['GT'])
    gt_nb_ids = os.path.join(data_folder[name]['ROOT'], data_folder[name]['GT_nb'])

    train_dict = {
        IMG: [[img_id.format(id) for img_id in img_ids] for id in train_id],
        GT: [gt_ids.format(id) for id in train_id],
        GTNB: [gt_nb_ids.format(id) for id in train_id]
    }

    val_dict = {
        IMG: [[img_id.format(id) for img_id in img_ids] for id in val_id],
        GT: [gt_ids.format(id) for id in val_id],
        GTNB: [gt_nb_ids.format(id) for id in val_id]
    }

    test_dict = {
        IMG: [[img_id.format(id) for img_id in img_ids] for id in test_ids],
        GT: [gt_ids.format(id) for id in test_ids],
        GTNB: [gt_nb_ids.format(id) for id in test_ids]
    }

    # fix some weird format files for DSM files
    if name == 'Potsdam':
        for i in range(len(train_dict[IMG])):
            for j in range(len(train_dict[IMG][i])):
                train_dict[IMG][i][j] = train_dict[IMG][i][j].replace('_7_normalized', '_07_normalized')
                train_dict[IMG][i][j] = train_dict[IMG][i][j].replace('_8_normalized', '_08_normalized')
                train_dict[IMG][i][j] = train_dict[IMG][i][j].replace('_9_normalized', '_09_normalized')

        for i in range(len(val_dict[IMG])):
            for j in range(len(val_dict[IMG][i])):
                val_dict[IMG][i][j] = val_dict[IMG][i][j].replace('_7_normalized', '_07_normalized')
                val_dict[IMG][i][j] = val_dict[IMG][i][j].replace('_8_normalized', '_08_normalized')
                val_dict[IMG][i][j] = val_dict[IMG][i][j].replace('_9_normalized', '_09_normalized')

        for i in range(len(test_dict[IMG])):
            for j in range(len(test_dict[IMG][i])):
                test_dict[IMG][i][j] = test_dict[IMG][i][j].replace('_7_normalized', '_07_normalized')
                test_dict[IMG][i][j] = test_dict[IMG][i][j].replace('_8_normalized', '_08_normalized')
                test_dict[IMG][i][j] = test_dict[IMG][i][j].replace('_9_normalized', '_09_normalized')

    return train_dict, val_dict, test_dict


def split_train_val_test_sets2(data_folder=DATA_FOLDER, name='Potsdam', bands=['RGB'], KF=None, k=1, seeds=69278):
    sub_ids = data_folder[name]['IDs_v2']  # this is for local hold-out test
    train_ids = [item for item in data_folder[name]['IDs']]
    test_ids = data_folder[name]['IDs_test']  # which is predefined, in stead of randomly selected
    fix_val = data_folder[name]['IDs_fv']

    train_ids = [item for item in train_ids if
                 item not in test_ids + sub_ids + fix_val]  # remove local test and hold-out test ids from list
    val_id = [item for item in fix_val]
    train_id, val_id = list(train_ids), list(val_id)

    if name == 'Vaihingen':
        if 'DSM' in bands:
            train_id.remove('11')  # area11 dsm file not valid, remove it from training id list

    print('train ids: ', train_id)
    print('val_id ids: ', val_id)

    img_ids = [os.path.join(data_folder[name]['ROOT'], data_folder[name][band]) for band in bands]
    gt_ids = os.path.join(data_folder[name]['ROOT'], data_folder[name]['GT'])
    gt_nb_ids = os.path.join(data_folder[name]['ROOT'], data_folder[name]['GT_nb'])

    train_dict = {
        IMG: [[img_id.format(id) for img_id in img_ids] for id in train_id],
        GT: [gt_ids.format(id) for id in train_id],
        GTNB: [gt_nb_ids.format(id) for id in train_id]
    }

    val_dict = {
        IMG: [[img_id.format(id) for img_id in img_ids] for id in val_id],
        GT: [gt_ids.format(id) for id in val_id],
        GTNB: [gt_nb_ids.format(id) for id in val_id]
    }

    test_dict = {
        IMG: [[img_id.format(id) for img_id in img_ids] for id in test_ids],
        GT: [gt_ids.format(id) for id in test_ids],
        GTNB: [gt_nb_ids.format(id) for id in test_ids]
    }

    # fix some weird format files for DSM files
    if name == 'Potsdam':
        for i in range(len(train_dict[IMG])):
            for j in range(len(train_dict[IMG][i])):
                train_dict[IMG][i][j] = train_dict[IMG][i][j].replace('_7_normalized', '_07_normalized')
                train_dict[IMG][i][j] = train_dict[IMG][i][j].replace('_8_normalized', '_08_normalized')
                train_dict[IMG][i][j] = train_dict[IMG][i][j].replace('_9_normalized', '_09_normalized')

        for i in range(len(val_dict[IMG])):
            for j in range(len(val_dict[IMG][i])):
                val_dict[IMG][i][j] = val_dict[IMG][i][j].replace('_7_normalized', '_07_normalized')
                val_dict[IMG][i][j] = val_dict[IMG][i][j].replace('_8_normalized', '_08_normalized')
                val_dict[IMG][i][j] = val_dict[IMG][i][j].replace('_9_normalized', '_09_normalized')

        for i in range(len(test_dict[IMG])):
            for j in range(len(test_dict[IMG][i])):
                test_dict[IMG][i][j] = test_dict[IMG][i][j].replace('_7_normalized', '_07_normalized')
                test_dict[IMG][i][j] = test_dict[IMG][i][j].replace('_8_normalized', '_08_normalized')
                test_dict[IMG][i][j] = test_dict[IMG][i][j].replace('_9_normalized', '_09_normalized')

    return train_dict, val_dict, test_dict


def convert_from_color(arr_3d, palette=None):
    if palette is None:
        palette = palette_org
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    invert_palette = {v: k for k, v in palette.items()}

    for c, i in invert_palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


class IsprsDataset(Dataset):
    def __init__(self, mode='train', file_lists=None, windSize=(256, 256),
                 num_samples=10000, cache=True, pre_norm=False):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.norm = pre_norm
        self.cache = cache
        self.winsize = windSize
        self.samples = num_samples

        # List of files
        self.image_files = file_lists[IMG]  # image_files = [[bands1, bands2,..], ...]
        self.mask_files = file_lists[GT]    # mask_files = [gt1, gt2, ...]

        # Sanity check : raise an error if some files do not exist
        for f in np.concatenate(self.image_files).ravel().tolist() + self.mask_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        if self.cache:
            self.image_cache_ = {}
            self.label_cache_ = {}

    def __len__(self):
        return self.samples

    def __getitem__(self, i):
        # Pick a random image
        # global image
        random_idx = random.randint(0, len(self.image_files) - 1)

        if random_idx in self.image_cache_.keys():
            image = self.image_cache_[random_idx]
        else:
            if len(self.image_files[random_idx]) > 1:
                imgs = []
                for i in range(len(self.image_files[random_idx])):
                    if i == 0:
                        img = imload(self.image_files[random_idx][i])
                        imgs.append(img)
                    else:
                        img = imload(self.image_files[random_idx][i], gray=True)
                        img = np.expand_dims(img, 2)
                        imgs.append(img)
                    image = np.concatenate(imgs, 2)
            else:
                image = imload(self.image_files[random_idx][0])

            # If the tile hasn't been loaded yet, put in cache
            if self.cache:
                self.image_cache_[random_idx] = image

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            label = imload(self.mask_files[random_idx])
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a randomly cropped patch
        image_p, label_p = img_mask_crop(image=image, mask=label, size=self.winsize, limits=self.winsize)

        # Data augmentation during training or validation
        if self.mode is 'train':
            image_p, label_p = self.train_augmentation(image_p, label_p)
        elif self.mode is 'val':
            image_p, label_p = self.val_augmentation(image_p, label_p)

        image_p = np.asarray(image_p, np.float32).transpose((2, 0, 1)) / 255.0
        label_p = np.asarray(convert_from_color(label_p), dtype='int64')

        image_p, label_p = torch.from_numpy(image_p), torch.from_numpy(label_p)

        if self.norm:
            image_p = self.normalize(image_p)

        return image_p, label_p

    @classmethod
    def train_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            ShiftScaleRotate(p=0.2, rotate_limit=10, scale_limit=0.1),

        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    @classmethod
    def val_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    @classmethod
    def normalize(cls, img):
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        norm = standard_transforms.Compose([standard_transforms.Normalize(*mean_std)])
        return norm(img)



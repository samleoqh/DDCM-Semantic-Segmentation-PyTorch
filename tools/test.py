from __future__ import division

from models.builder import load_model

from datasets.isprs.isprs_loader import *
from datasets.isprs.isprs_configs import IsprsConfigs
from utils.visual_functions import *

import torchvision.transforms as st
from sklearn.metrics import confusion_matrix
from skimage import io
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd

import time
import itertools


test_args = IsprsConfigs(net_name= 'ddcm_r50', #'ddcm_core',  # 'DDCM_SER50',
                         data='Vaihingen',  # Potsdam, Vaihingen
                         bands_list=['IRRG'],    # RGB, IRRG, 'DSM'
                         note='test_output_ddcm_r50'
                         )

test_args.snapshot = '/home/liu/Desktop/git/DDCM/weights/Vaihingen_epoch_116.pth'

test_args.input_size = [448, 448]  # [448, 448]
test_args.val_batch = 12
test_args.pre_norm = False


def loadtestimg(image_files):
    for k in range(len(image_files)):
        if len(image_files[k]) > 1:
            imgs = []
            for i in range(len(image_files[k])):
                if i == 0:
                    # for the first band, read as RGB
                    img = imload(image_files[k][i])
                    imgs.append(img)
                else:
                    # for other bands, read as to gray [0, 255]
                    img = imload(image_files[k][i], gray=True)
                    img = np.expand_dims(img, 2)  # expand dims to h x w x 1
                    imgs.append(img)
                image = np.concatenate(imgs, 2)
        else:
            image = imload(image_files[k][0])
        yield image


def main():
    output_path = os.path.join(test_args.save_path, 'test_outputs')
    check_mkdir(output_path)
    net = load_model(name=test_args.model, classes=test_args.nb_classes).cuda()

    net.load_state_dict(torch.load(test_args.snapshot))

    net.eval()

    _, _, test_files = test_args.get_file_list()

    _, all_preds, all_gts, cm = tta_test(net, stride=100, batch_size=test_args.val_batch,
                                         norm=test_args.pre_norm, nb_gt=True,
                                         window_size=test_args.input_size, labels=test_args.labels,
                                         test_set=test_files, all=True)

    scores_filename = '{}_{}_ep37_448s100_tta_nb_real_test.csv'.format(test_args.model, test_args.dataset)

    scores = compute_scores(cm, test_args.labels)
    scores.to_csv(os.path.join(output_path, scores_filename))

    for p, id_ in zip(all_preds, DATA_FOLDER[test_args.dataset]['IDs_test']):
        img = convert_to_color(p, test_args.palette)
        filename = './{}_inference_tile{}_ep37_448s100_tta.png'.format(
            test_args.model, id_)
        io.imsave(os.path.join(output_path, filename), img)


def tta_test(net, all=False, labels=LABELS, norm=False, nb_gt=True,
             test_set=None, stride=256, batch_size=5, window_size=(256, 256)):
    test_images = (loadtestimg(test_set[IMG]))
    test_labels = (np.asarray(io.imread(label), dtype='uint8')
                   for label in test_set[GT])
    eroded_labels = (convert_from_color(io.imread(label_nb))
                     for label_nb in test_set[GTNB])
    all_preds = []
    all_gts = []
    num_class = len(labels)

    for img, gt, gt_nb in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_set[GTNB]), leave=False):

        pred = np.zeros(img.shape[:2] + (num_class,))
        img = np.asarray(img, dtype='float32')
        img = st.ToTensor()(img)
        img = img / 255.0
        if norm:
            img = st.Normalize(*mean_std)(img)
        img = img.cpu().numpy().transpose((1, 2, 0))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size

        stime = time.time()

        with torch.no_grad():
            for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):

                image_patches = [
                    np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                imgs_flip = [patch[:, ::-1, :] for patch in image_patches]
                imgs_mirror = [patch[:, :, ::-1] for patch in image_patches]

                image_patches = np.concatenate(
                    (image_patches, imgs_flip, imgs_mirror), axis=0)
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).cuda()

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                b, _, _, _ = outs.shape

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs[0:b // 3, :, :, :], coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                for out, (x, y, w, h) in zip(
                        outs[b // 3:2 * b // 3, :, :, :], coords):
                    out = out[:, ::-1, :]  # flip back
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                for out, (x, y, w, h) in zip(
                        outs[2 * b // 3: b, :, :, :], coords):
                    out = out[:, :, ::-1]  # mirror back
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                del (outs)

        print('inference cost time: ', time.time() - stime)

        pred = np.argmax(pred, axis=-1)

        gtc = convert_from_color(gt)
        all_preds.append(pred)

        if nb_gt:
            all_gts.append(gt_nb)
        else:
            all_gts.append(gtc)

    accuracy, cm = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts, cm
    else:
        return accuracy


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))

    print("Confusion matrix :")
    print(cm)

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except BaseException:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    return accuracy, cm


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def compute_scores(conmatrix=None, col_idx=None, savefile=None):
    if col_idx is None:
        col_idx = ['Building', 'Tree', 'Low-vege', 'Clutter', 'Surface', 'Car']

    row_idx = ['Classes',
               'TP', 'TN', 'FP', 'FN',
               'specificity', 'precision', 'recall', 'f1_score',
               'jaccard', 'dicesim', 'randacc', 'arearoc']

    scores = pd.DataFrame(columns=col_idx, index=row_idx)
    scores.loc['Classes'] = col_idx
    scores.columns = scores.iloc[0]
    scores = scores.reindex(scores.index.drop('Classes'))

    col_sum = conmatrix.sum(axis=0)  # number of estimated classes
    row_sum = conmatrix.sum(axis=1)  # number of true/reference classes
    tol_sum = conmatrix.sum()    # total pixels, a scalar

    M = len(conmatrix)
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    for i in range(M):
        tp[i] = conmatrix[i, i]
        fp[i] = np.sum(conmatrix[:, i]) - tp[i]
        fn[i] = np.sum(conmatrix[i, :]) - tp[i]

    tn = tol_sum - fp - row_sum

    # accuracy = sum([conmatrix[x][x] for x in range(len(conmatrix))]) / tol_sum

    specificity = tn / (tol_sum - row_sum)
    precision = tp / (tp + fp)              # = tp/col_sum
    recall = tp / (tp + fn)
    f1_score = 2 * recall * precision / (recall + precision)

    jaccard = tp / (tp + fp + fn)
    dicesim = 2 * tp / (col_sum + row_sum)
    randacc = (tp + tn) / tol_sum
    arearoc = (tp / row_sum + tn / (tol_sum - row_sum)) / 2

    scores.loc['TP'] = tp
    scores.loc['TN'] = tn
    scores.loc['FP'] = fp
    scores.loc['FN'] = fn
    scores.loc['specificity'] = np.around(specificity, 4)
    scores.loc['precision'] = np.around(precision, 4)
    scores.loc['recall'] = np.around(recall, 4)
    scores.loc['f1_score'] = np.around(f1_score, 4)
    scores.loc['jaccard'] = np.around(jaccard, 4)
    scores.loc['dicesim'] = np.around(dicesim, 4)
    scores.loc['randacc'] = np.around(randacc, 4)
    scores.loc['arearoc'] = np.around(arearoc, 4)

    mean_score = scores.drop(labels=['Clutter'], axis=1)  # drop out clutter
    # np.around(mean_score.mean(axis=1), 4)
    scores['Avg1'] = mean_score.mean(axis=1)
    scores['Avg'] = scores.mean(axis=1)  # np.around(scores.mean(axis=1), 4)

    return scores


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()

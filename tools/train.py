from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import torchvision.utils as vutils
from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter


from models.builder import load_model
from models.loss_functions.ce_loss import CrossEntropy2D
from utils.lr_functions import init_params_lr, adjust_learning_rate
from utils.eval_matrix import AverageMeter, evaluate
from utils.visual_functions import *

from datasets.isprs.isprs_loader import *
from datasets.isprs.isprs_configs import IsprsConfigs

from torch.utils.data import DataLoader


def setup_train_args():
    train_args = IsprsConfigs(net_name='ddcm_r50', data='Vaihingen',  # 'Potsdam',  # 'Vaihingen',
                              bands_list=['IRRG'],
                              kf=1,
                              note='debug_rep')

    train_args.input_size = [256, 256]
    train_args.val_size = [448, 448]  # [256, 256]
    train_args.train_samples = 5000
    train_args.train_batch = 5

    train_args.lr = 8.5e-5 / np.sqrt(2.0)
    train_args.steps = 15
    train_args.gamma = 0.85
    train_args.weight_decay = 2e-5
    train_args.snapshot = ''

    train_args.write2txt()
    return train_args


def random_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


train_args = setup_train_args()
writer = SummaryWriter(os.path.join(train_args.save_path, 'tblog'))
visualize, restore = get_visualize(train_args)


def main():
    random_seed(train_args.seeds)
    net = load_model(name=train_args.model, classes=train_args.nb_classes, load_weights=False, skipp_layer=None).cuda()

    net, start_epoch = train_args.resume_train(net)
    net.train()

    # prepare dataset for training and validation
    train_set, val_set = train_args.get_dataset()
    train_loader = DataLoader(dataset=train_set, batch_size=train_args.train_batch, num_workers=0, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=train_args.val_batch, num_workers=0)

    mfb_weight = torch.from_numpy(np.asarray(train_args.weights, dtype=np.float32)).cuda()
    criterion = CrossEntropy2D(weight=mfb_weight).cuda()
    params = init_params_lr(net, train_args)
    optimizer = optim.Adam(params, amsgrad=True)
    lr_scheduler = StepLR(optimizer, step_size=train_args.steps, gamma=train_args.gamma)
    new_ep = 0

    while True:
        train_main_loss = AverageMeter()

        train_args.lr = optimizer.param_groups[0]['lr']

        num_iter = len(train_loader)
        curr_iter = ((start_epoch + new_ep) - 1) * num_iter
        print('---curr_iter: {}, num_iter per epoch: {}---'.format(curr_iter, num_iter))

        starttime = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)

            optimizer.zero_grad()
            outputs = net(inputs)

            main_loss = criterion(outputs, labels)

            loss = main_loss

            loss.backward()
            optimizer.step()

            adjust_learning_rate(optimizer, curr_iter, train_args)
            train_main_loss.update(main_loss.item(), N)

            curr_iter += 1
            writer.add_scalar('main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], curr_iter)

            if (i + 1) % train_args.print_freq == 0:
                process_time = time.time() - starttime
                print('[epoch %d], [iter %d / %d], [loss %.5f], [lr %.10f], [time %.3f]' %
                      (start_epoch + new_ep, i + 1, num_iter, train_main_loss.avg,
                       optimizer.param_groups[0]['lr'], process_time))

                starttime = time.time()

        validate(net, val_loader, criterion, optimizer, start_epoch + new_ep, new_ep)

        lr_scheduler.step(epoch=(start_epoch + new_ep))
        new_ep += 1


def validate(net, val_loader, criterion, optimizer, epoch, new_ep):
    net.eval()
    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    with torch.no_grad():
        for vi, (inputs, gts) in enumerate(val_loader):
            inputs, gts = inputs.cuda(), gts.cuda()
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)
            outputs = net(inputs)

            predictions = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            val_loss.update(criterion(outputs, gts).item(), N)
            if random.random() > train_args.save_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(inputs.data.squeeze(0).cpu())
            gts_all.append(gts.data.squeeze(0).cpu().numpy())
            predictions_all.append(predictions)

    update_ckpt(net, optimizer, epoch, new_ep, val_loss,
                inputs_all, gts_all, predictions_all)

    net.train()
    return val_loss, inputs_all, gts_all, predictions_all


def update_ckpt(net, optimizer, epoch, new_ep, val_loss,
                inputs_all, gts_all, predictions_all):
    avg_loss = val_loss.avg

    acc, acc_cls, mean_iu, fwavacc, f1 = evaluate(predictions_all, gts_all, train_args.nb_classes)

    writer.add_scalar('val_loss', avg_loss, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('f1_score', f1, epoch)

    updated = train_args.update_best_record(epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1)

    val_visual = []

    snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_f1_%.5f_lr_%.10f' % (
        epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1, optimizer.param_groups[0]['lr']
    )

    if updated or (train_args.best_record['val_loss'] > avg_loss):
        torch.save(net.state_dict(), os.path.join(train_args.save_path, snapshot_name + '.pth'))
    if updated or (new_ep % 5 == 0):
        val_visual = visual_ckpt(epoch, new_ep, inputs_all, gts_all, predictions_all)

    if len(val_visual) > 0:
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)


def visual_ckpt(epoch, new_ep, inputs_all, gts_all, predictions_all):
    val_visual = []
    if train_args.save_pred:
        to_save_dir = os.path.join(train_args.save_path, str(epoch) + '_' + str(new_ep))
        check_mkdir(to_save_dir)

    for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
        if data[0] is None:
            continue

        if train_args.val_batch == 1:
            input_pil = restore(data[0][0:3, :, :]) # only for the first 3 bands
            gt_pil = colorize_mask(data[1], train_args.palette)
            predictions_pil = colorize_mask(data[2], train_args.palette)
        else:
            input_pil = restore(data[0][0][0:3, :, :])  # only for the first 3 bands
            gt_pil = colorize_mask(data[1][0], train_args.palette)
            predictions_pil = colorize_mask(data[2][0], train_args.palette)

        if train_args.save_pred:
            input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
            predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
            gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))

        val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                           visualize(predictions_pil.convert('RGB'))])
    return val_visual


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()

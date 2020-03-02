from __future__ import print_function, division


def init_params_lr(net, opt):
    bias_params = []
    nonbias_params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                nonbias_params.append(value)
    params = [
        {'params': nonbias_params,
         'lr': opt.lr,
         'weight_decay': opt.weight_decay},
        {'params': bias_params,
         'lr': opt.lr * 2.0,
         'weight_decay': 0}
    ]
    return params


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, opt):
    lr = lr_poly(opt.lr, i_iter, opt.max_iter, opt.lr_decay)
    optimizer.param_groups[0]['lr'] = lr  # weights
    optimizer.param_groups[0]['weight_decay'] = opt.weight_decay
    optimizer.param_groups[1]['lr'] = lr * 2.0  # bias
    return lr

from email.policy import strict
import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional


def save_model(net, optim, scheduler, recorder, is_best=False):
    model_dir = os.path.join(recorder.work_dir, 'ckpt')
    os.system('mkdir -p {}'.format(model_dir))
    epoch = recorder.epoch
    ckpt_name = 'best' if is_best else 'latest'
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(ckpt_name)))

    # remove previous pretrained model if the number of models is too big
    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    # if len(pths) <= 2:
    #     return
    # os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network_specified(net, model_dir, logger=None):
    pretrained_net = torch.load(model_dir)['net']
    net_state = net.state_dict()
    state = {}
    for k, v in pretrained_net.items():
        if k not in net_state.keys() or v.size() != net_state[k].size():
            if logger:
                logger.info('skip weights: ' + k)
            continue
        state[k] = v
    net.load_state_dict(state, strict=False)


def load_network(net,optim,scheduler,recorder,model_dir, finetune_from=None, logger=None, cfg=None):
    if finetune_from:
        if logger:
            logger.info('Finetune model from: ' + finetune_from)
        load_network_specified(net, finetune_from, logger)
        return
    pretrained_model = torch.load(model_dir)
    net.load_state_dict(pretrained_model['net'], strict=True)
    load_network_specified(net, model_dir, logger)
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    recorder.epoch = pretrained_model['epoch']
    # os.system('rm -rf '+ recorder.work_dir)
    if not cfg.validate:
        recorder.work_dir = os.path.join(*model_dir.split('/')[:-2])
        recorder.log_path = os.path.join(recorder.work_dir, 'run_info.log')
        from .logger import init_logger
        init_logger(recorder.log_path)
        from torch.utils.tensorboard import SummaryWriter
        recorder.tensorboard_writer = SummaryWriter(os.path.join(recorder.work_dir,'tensorboard'))
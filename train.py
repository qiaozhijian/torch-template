import os
import numpy as np
import argparse
import glob
from common_utils.config import cfg
from common_utils.prepare_utils import load_config
import tqdm
from common_utils.optimization import build_scheduler
from training.train_utils import train_one_epoch
from common_utils.model_utils import save_checkpoint, checkpoint_state, load_checkpoint

import torch
from training.loss_func.simple_loss import SimpleLoss

from training.dataloaders import make_dataloaders
from models import get_model

def train(cfg, logger, writer):

    train_loader, val_loader = make_dataloaders(cfg)
    model = get_model(cfg)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.LR, weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY)

    start_it, start_epoch, last_epoch = load_checkpoint(model, optimizer, logger, True)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(optimizer, len(train_loader), cfg.OPTIMIZATION.NUM_EPOCHS, last_epoch, cfg.OPTIMIZATION)

    loss_fn = SimpleLoss(cfg)

    accumulated_iter = start_it

    if cfg.LOCAL_RANK == 0:
        tbar = tqdm.trange(start_epoch, cfg.OPTIMIZATION.NUM_EPOCHS, desc='epochs', dynamic_ncols=True)

    for epoch in range(start_epoch, cfg.OPTIMIZATION.NUM_EPOCHS):

        accumulated_iter = train_one_epoch(
            epoch, model, optimizer, train_loader, accumulated_iter, logger, loss_fn, writer)

        if cfg.LOCAL_RANK == 0:
            cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
            disp_dict = {'lr': cur_lr}
            tbar.set_postfix(disp_dict)
            tbar.update()

        if cfg.OPTIMIZATION.OPTIMIZER != 'adam_onecycle':
            lr_scheduler.step()
        # save trained model
        trained_epoch = epoch + 1
        if trained_epoch % 1 == 0 and cfg.LOCAL_RANK == 0:

            ckpt_list = glob.glob(str(cfg.CKPT_DIR / 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)

            if ckpt_list.__len__() >= 5:
                for cur_file_idx in range(0, len(ckpt_list) - 5 + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name = cfg.CKPT_DIR / ('checkpoint_epoch_%d' % trained_epoch)
            save_checkpoint(
                checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for demo')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--pretrained_model', type=str, default=None)

    args = parser.parse_args()
    logger, writer = load_config(args)

    train(cfg, logger, writer)

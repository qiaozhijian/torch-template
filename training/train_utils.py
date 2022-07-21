from time import time
from common_utils.config import cfg
import torch
import numpy as np
import tqdm

def train_one_epoch(epoch, model, opt, train_loader, accumulated_iter,
                    logger, loss_fn, writer):

    if cfg.LOCAL_RANK == 0:
        pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train', dynamic_ncols=True)

    model.train()
    for bi, batch in enumerate(train_loader):
        # get_semantic_map：非语义类，divider，ped_crossing，boundary
        t0 = time()
        opt.zero_grad()
        accumulated_iter += 1

        batch = batch2cuda(batch)
        pred = model(batch)

        loss = loss_fn(pred, 0)
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)

        opt.step()
        t1 = time()

        if cfg.LOCAL_RANK == 0:
            pbar.update()
            disp_dict = {
                "time": t1 - t0
            }
            pbar.set_postfix(disp_dict)

            writer.add_scalar('train/step_time', t1 - t0, accumulated_iter)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    if cfg.LOCAL_RANK == 0:
        pbar.close()

    return accumulated_iter


def batch2cuda(batch):
    return batch
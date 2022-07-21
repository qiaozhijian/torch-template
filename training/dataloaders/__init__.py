import torch
from .simple_dataset import SimpleDataset

def make_dataloaders(cfg):

    bsz = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    bsz_val = cfg.OPTIMIZATION.TEST_BATCH_SIZE_PER_GPU
    nworkers = cfg.DATA_CONFIG.NUM_WORKERS

    train_dataset = SimpleDataset(cfg, is_train=True)
    val_dataset = SimpleDataset(cfg, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz_val, shuffle=False, num_workers=nworkers)

    return train_loader, val_loader
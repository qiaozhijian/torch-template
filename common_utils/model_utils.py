from common_utils.config import cfg
import torch
import os
import glob

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        version = torch.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state,
            'batch_size': cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            'optimizer_state': optim_state, 'version': version}

def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
    
def load_checkpoint(model, optimizer = None, logger = None, to_cpu = False):
    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    # 加载预训练模型
    if cfg.PRETRAIN_MODEL is not None:
        load_params_from_file(model, filename=cfg.PRETRAIN_MODEL, to_cpu=to_cpu, logger=logger)

    if optimizer is None:
        return

    # 加载checkpoint
    if cfg.OPTIMIZATION.CHECKPOINT is not None:
        it, last_epoch = model.load_params_with_optimizer(model, cfg.OPTIMIZATION.CHECKPOINT, to_cpu=to_cpu, optimizer=optimizer, logger=logger)
        start_epoch = last_epoch + 1
    else:
        ckpt_list = glob.glob(str(cfg.CKPT_DIR / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, last_epoch = load_params_with_optimizer(model,
                ckpt_list[-1], to_cpu=to_cpu, optimizer=optimizer, logger=logger
            )
            start_epoch = last_epoch + 1

    if start_epoch == cfg.OPTIMIZATION.NUM_EPOCHS and logger is not None:
        logger.info("No more epoch needs to be trained!!!")

    return it, start_epoch, last_epoch
            
def load_params_from_file(model, filename, logger, to_cpu=False):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    model_state_disk = checkpoint['model_state']

    version = checkpoint.get("version", None)
    if version is not None and logger is not None:
        logger.info('==> Checkpoint trained from version: %s' % version)

    model.load_state_dict(model_state_disk, strict=True)

def load_params_with_optimizer(model, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    epoch = checkpoint.get('epoch', -1)
    it = checkpoint.get('it', 0.0)

    load_params_from_file(model, filename=filename, to_cpu=to_cpu, logger=logger)

    if optimizer is not None:
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            assert filename[-4] == '.', filename
            src_file, ext = filename[:-4], filename[-3:]
            optimizer_filename = '%s_optim.%s' % (src_file, ext)
            if os.path.exists(optimizer_filename):
                optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

    if 'version' in checkpoint:
        print('==> Checkpoint trained from version: %s' % checkpoint['version'])
    logger.info('==> Done')

    return it, epoch
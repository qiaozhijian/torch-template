import os
from pathlib import Path
from common_utils.config import cfg, cfg_from_yaml_file
from common_utils import common_utils
import datetime
import socket
from tensorboardX import SummaryWriter

def load_config(args, evaluate = False):
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.PRETRAIN_MODEL = args.pretrained_model

    common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'outputs' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.CKPT_DIR = output_dir / 'ckpt'
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    cfg.LOGS_DIR = output_dir / 'logs'
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cfg.EVAL_DIR = output_dir / 'eval'
    cfg.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    cfg.EVAL_IMG_DIR = output_dir / 'eval' / 'imgs'
    cfg.EVAL_IMG_DIR.mkdir(parents=True, exist_ok=True)

    os.system('cp {} {}'.format(args.cfg_file, output_dir / '{}.yaml'.format(cfg.TAG)))
    log_file = cfg.LOGS_DIR / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('Load configuration from {}.'.format(args.cfg_file))

    # 本地调试，多线程置为1
    if 'Lenovo' in socket.gethostname():
        cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU = 1
        cfg.OPTIMIZATION.TEST_BATCH_SIZE_PER_GPU = 1
        cfg.DATA_CONFIG.NUM_WORKERS = 0
        cfg.DATA_CONFIG.DATASET_FOLDER = "/media/qzj/Extreme SSD/datasets/nuscenes"
        cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
        cfg.DATA_CONFIG.EVAL_SET = 'mini_val'

    if not evaluate:
        writer_dir = output_dir / 'writer'
        writer_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(writer_dir)
        return logger, writer
    else:
        return logger
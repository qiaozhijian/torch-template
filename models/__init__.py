from .simplenet import SimpleNet
import logging

def get_model(cfg):
    method = cfg.MODEL.NAME

    if method.lower() == 'SimpleNet'.lower():
        model = SimpleNet()
    else:
        raise NotImplementedError

    logger = logging.getLogger('My_project')
    if cfg.LOCAL_RANK == 0:
        if hasattr(model, 'print_info'):
            model.print_info(logger)
        else:
            n_params = sum([param.nelement() for param in model.parameters()])
            logger.info('Model {} : params: {:4f}M'.format(method, n_params * 4 / 1000 / 1000))

    return model

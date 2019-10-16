import math


def get_learning_rate(cfg, iter_num):
    base_lr = cfg.LEARNING_RATE
    policy = cfg.LEARNING_RATE_POLICY
    lr_decay = cfg.LEARNING_RATE_DECAY
    lr_step = cfg.LEARNING_RATE_STEP
    lr_stepvalues = cfg.LEARNING_RATE_STEPVALUES
    lr_gamma = cfg.LEARNING_RATE_GAMMA
    lr_power = cfg.LEARNING_RATE_POWER

    if policy == 'fixed':
        return base_lr
    elif policy == 'step' and lr_decay is not None and lr_step is not None:
        return base_lr * math.pow(lr_decay, iter_num // lr_step)
    elif policy == 'exp' and lr_gamma is not None:
        return base_lr * pow(lr_gamma, iter_num)
    elif policy == 'multisteps' and lr_stepvalues is not None and lr_decay is not None:
        lr = base_lr
        for s in lr_stepvalues:
            if s < iter_num:
                lr = lr * lr_decay
        return lr
    elif policy == 'inv' and lr_gamma is not None and lr_power is not None:
        return base_lr * math.pow(1 + lr_gamma * iter_num, -lr_power)
    else:
        return base_lr

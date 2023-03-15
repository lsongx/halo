import mmcv
from mmcv.runner import Hook


class LrIterHook(Hook):
    """Adjust the param of a dataset (by iterations).
    """

    def __init__(self,
                 optimizer,
                 lr_config,
                 start_iter,
                 end_iter,
                 logger=None):
        self.optimizer = optimizer
        self.lr_config = lr_config
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.logger = logger

    def after_train_iter(self, runner):
        if (runner.iter+1) == self.start_iter:
            # self.ori_optimizer = runner.optimizer
            # optimizer = mmcv.runner.build_optimizer(runner.model, self.optimizer)
            # runner.optimizer = optimizer
            for idx, h in enumerate(runner._hooks):
                if isinstance(h, mmcv.runner.hooks.lr_updater.LrUpdaterHook):
                    self.logger.info(f'Hook {runner._hooks[idx]} deleted')
                    self.ori_lr = runner._hooks[idx]
            lr_config = mmcv.build_from_cfg(self.lr_config, mmcv.runner.hooks.HOOKS)
            lr_config.before_run(runner)
            runner.register_lr_hook(lr_config)
            self.logger.info(f'Current hooks:\n {runner._hooks}')

        if (runner.iter+1) == self.end_iter:
            # runner.optimizer = self.ori_optimizer
            for idx, h in enumerate(runner._hooks):
                if isinstance(h, mmcv.runner.hooks.lr_updater.LrUpdaterHook):
                    self.logger.info(f'Hook {runner._hooks[idx]} deleted')
                    runner._hooks[idx] = self.ori_lr
            self.logger.info(f'Current hooks:\n {runner._hooks}')

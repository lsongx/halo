from json import load
from mmcv.runner import Hook


class IterAdjustHook(Hook):
    """Adjust the param of a dataset (by iterations).
    """

    def __init__(self, logger=None):
        self.logger = logger

    def after_train_iter(self, runner):
        if not hasattr(runner.model.module, 'iter'):
            self.logger.info(f'Create iter attribute for {runner.model.module}')

        if hasattr(runner.model.module.iter, 'data'):
            runner.model.module.iter *= 0
            runner.model.module.iter += runner.iter # it is a nn parameter
        else:
            runner.model.module.iter = runner.iter

        for loader in runner.data_loaders:
            if hasattr(loader.dataset, 'dataset'):
                # handle warpper
                dataset = loader.dataset.dataset
            else:
                dataset = loader.dataset
            if runner.iter == 0:
                if loader.num_workers > 0:
                    self.logger.warning(f'iter not used for {dataset} since num_workers>0')
                    return
            if not hasattr(dataset, 'iter'):
                self.logger.info(f'Create iter attribute for {dataset}')
            dataset.iter = runner.iter


from matplotlib.pyplot import isinteractive
import torch
from json import load
from mmcv.runner import Hook


class UpdateModelByDataset(Hook):
    """Adjust the param of a dataset (by iterations).
    """

    def __init__(self, logger=None):
        self.logger = logger

    def before_run(self, runner):
        pass


class EPIUpdateModelByDataset(UpdateModelByDataset):
    def __init__(self, logger, grid_num=4, range_multi=1):
        super().__init__(logger=logger)
        self.grid_num = grid_num
        self.range_multi = range_multi

    def before_run(self, runner):
        for loader in runner.data_loaders:
            if hasattr(loader.dataset, 'dataset'):
                # handle warpper
                dataset = loader.dataset.dataset
            else:
                dataset = loader.dataset

        u_range = (dataset.u_max-dataset.u_min)/self.grid_num*self.range_multi/dataset.scale
        v_range = (dataset.v_max-dataset.v_min)/self.grid_num*self.range_multi/dataset.scale
        s_range = (dataset.s_max-dataset.s_min)/self.grid_num*self.range_multi/dataset.scale
        t_range = (dataset.t_max-dataset.t_min)/self.grid_num*self.range_multi/dataset.scale
        runner.model.module.max_moving_dis[0] *= max(u_range,v_range)
        runner.model.module.max_moving_dis[1] *= max(s_range,t_range)
        runner.model.module.min_moving_dis[0] *= min(u_range,v_range)
        runner.model.module.min_moving_dis[1] *= min(s_range,t_range)
        self.logger.info(f'max_moving_dis updated as {runner.model.module.max_moving_dis}')

        max_uv = runner.model.module.max_moving_dis[0]
        min_uv = runner.model.module.min_moving_dis[0]
        pm = runner.model.module.pixel_move
        def update():
            self.logger.info(
                f'near updated from {runner.model.module.near} to {near}; '
                f'far updated from {runner.model.module.far} to {far}')
            runner.model.module.far *= 0
            runner.model.module.far += far
            runner.model.module.near *= 0
            runner.model.module.near += near
        if isinstance(pm, (int, float)):
            if pm > 0:
                pm /= dataset.scale
                near = torch.atan((-pm)/max_uv)
                far = torch.atan((pm)/min_uv)
                update()
        elif isinstance(pm, list):
            pm_min = pm[0]
            pm_max = pm[1]
            uv_move = u_range
            st_move_min = pm_min/dataset.scale
            st_move_max = pm_max/dataset.scale
            near = torch.atan(st_move_min/uv_move)
            far = torch.atan(st_move_max/uv_move)
            update()
        else:
            raise NotImplementedError

        device = runner.model.module.center_uv.device
        if hasattr(runner.model.module, 'center_uv'):
            runner.model.module.center_uv += torch.tensor(dataset.center_uv, device=device)
            self.logger.info(f'center_uv updated to {runner.model.module.center_uv}')


class RangeEPIUpdateModelByDataset(UpdateModelByDataset):
    def __init__(self, logger):
        super().__init__(logger=logger)

    def before_run(self, runner):
        for loader in runner.data_loaders:
            if hasattr(loader.dataset, 'dataset'):
                # handle warpper
                dataset = loader.dataset.dataset
            else:
                dataset = loader.dataset

        u_range = (dataset.u_max-dataset.u_min)/dataset.scale
        v_range = (dataset.v_max-dataset.v_min)/dataset.scale
        s_range = (dataset.s_max-dataset.s_min)/dataset.scale
        t_range = (dataset.t_max-dataset.t_min)/dataset.scale
        runner.model.module.max_moving_dis[0] *= max(u_range,v_range)
        runner.model.module.max_moving_dis[1] *= max(s_range,t_range)
        self.logger.info(f'max_moving_dis updated as {runner.model.module.max_moving_dis}')


class SphereUpdateModelByDataset(Hook):
    def __init__(self, logger=None):
        self.logger = logger

    def before_run(self, runner):
        for loader in runner.data_loaders:
            if hasattr(loader.dataset, 'dataset'):
                # handle warpper
                dataset = loader.dataset.dataset
            else:
                dataset = loader.dataset
        
        runner.model.module.rad0 = dataset.rad0
        runner.model.module.rad1 = dataset.rad1


class LFPlaneUpdateModelByDataset(Hook):
    def __init__(self, logger=None):
        self.logger = logger

    def before_run(self, runner):
        for loader in runner.data_loaders:
            if hasattr(loader.dataset, 'dataset'):
                # handle warpper
                dataset = loader.dataset.dataset
            else:
                dataset = loader.dataset

        device = runner.model.module.uv_plane['normal'].device
        for k,v in dataset.uv_plane.items():
            runner.model.module.uv_plane[k] += torch.tensor(dataset.uv_plane[k], device=device)
            runner.model.module.st_plane[k] += torch.tensor(dataset.st_plane[k], device=device)

        if hasattr(runner.model.module, 'center_uv'):
            runner.model.module.center_uv += torch.tensor(dataset.center_uv, device=device)
            self.logger.info(f'center_uv updated to {runner.model.module.center_uv}')
        # runner.model.module.uv_plane = torch.nn.ParameterDict(
        #     {k: torch.nn.Parameter(torch.tensor(v), requires_grad=False) 
        #     for k,v in dataset.uv_plane.items()}
        # )
        # runner.model.module.st_plane = torch.nn.ParameterDict(
        #     {k: torch.nn.Parameter(torch.tensor(v), requires_grad=False) 
        #      for k,v in dataset.st_plane.items()}
        # )

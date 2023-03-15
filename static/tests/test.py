from vcnerf.datasets.loader.blender_loader import BlenderLoader
from vcnerf.datasets.nerf_dataset import NeRFDataset


split = 'train'
root_dir = '/home/sliu/code/blender-data/blender-output/sit_lauGh_static_19cam_360x480_24x32_focal50.0_64spl_step1/'

loader = BlenderLoader(root_dir)
# dataset = NeRFDataset(loader, split, holdout=4)
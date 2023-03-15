import torch
from vcnerf.models.renderer.dynamic_inverse import rot_from_axisangle
from scipy.spatial.transform import Rotation

# mat = rot_from_axisangle(torch.tensor([30,45,23]).float().view([1,1,3]))
mat = rot_from_axisangle(torch.tensor([1.4,1.7,0.4]).float().view([1,1,3]))
r = Rotation.from_matrix(mat.numpy()[0][:3,:3])
print(r.as_rotvec())
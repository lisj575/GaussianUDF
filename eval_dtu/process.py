from pytorch3d.structures import Meshes
import trimesh
import torch
import sys
sys.path.append("..")
from np_field import NPullNetwork
import os

path = '/data/lisj/2dgs_recon/output/20240628-udf_lr1e-3/'
mesh = trimesh.load(path + 'mesh/culled_mesh1.ply')
vertices = torch.from_numpy(mesh.vertices).cuda().float()
faces = torch.from_numpy(mesh.faces).cuda().long()
meshes = Meshes(verts=[vertices],
                faces=[faces])

normals = []
udf_field = NPullNetwork(range_=1.3).cuda()
ckpt, iter = torch.load(path+'field30000.pth') 
udf_field.load_state_dict(ckpt)
total_num = vertices.shape[0]
for i in range(0, total_num, 5000):
    point_n = udf_field.gradient(vertices[i:i+5000]).squeeze(1)
    point_n = torch.nn.functional.normalize(point_n, dim=-1)
    normals.append(point_n.detach())
normals = torch.cat(normals, dim=0)

vertices = vertices + normals*0.005
mesh.vertices = vertices.cpu().numpy()
mesh.export(path + 'mesh/culled_mesh.ply')

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import mcubes
import os      
import trimesh
from extract_mesh import get_mesh_udf_fast
# from extract_mesh import get_mesh_udf_fast
#from DualMeshUDF import extract_mesh

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class UDFNetwork(torch.nn.Module):
    def __init__(self,
                 max_batchsize=5000,
                 #dims=[3, 512, 512, 512, 512, 512, 512, 512, 512, 1],
                 dims=None,
                 skip_in=(4,),
                 multires=6,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False, range_=2.0, last_layer='abs'):
        super(UDFNetwork, self).__init__()
        if dims is None:
            dims=[3, 256, 256, 256, 256, 256, 256, 256, 256, 1]
        self.max_batchsize = max_batchsize
        self.embed_fn_fine = None
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims[0])
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.range_ = range_

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            
            #print(dims[l], out_dim, l)
            lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=2*np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.000001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-2*np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.000001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.dropout = torch.nn.Dropout(0.2)
        self.activation = torch.nn.ReLU()
        self.last_layer = last_layer
        assert self.last_layer in ['relu', 'abs']
      
        

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        if self.last_layer == 'relu':
            return self.activation(x) / self.scale
        elif self.last_layer == 'abs':
            return abs(x) / self.scale
        
    def nudf_extract(self, resolution=128, base_exp_dir='', dist_threshold_ratio=2.0, threshold=0.0, iteration=0, postfix='', voxel_origin=[-0.55,-0.55,-0.55]):
        func = lambda pts: self.sdf(pts)
        func_grad = lambda pts: self.gradient(pts).squeeze(1)
        try:
            pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                            indices=None, N_MC=resolution,
                                                                            gradient=True, eps=0.005,
                                                                            border_gradients=True,
                                                                            smooth_borders=True,
                                                                            dist_threshold_ratio=dist_threshold_ratio,
                                                                            level=threshold, voxel_origin=voxel_origin)
        except:
            pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                            indices=None, N_MC=resolution,
                                                                            gradient=True, eps=0.002,
                                                                            border_gradients=False,
                                                                            smooth_borders=False,
                                                                            dist_threshold_ratio=dist_threshold_ratio,
                                                                            level=threshold, voxel_origin=voxel_origin)
        
        os.makedirs(os.path.join(base_exp_dir, 'mesh_udf'), exist_ok=True)
        pred_mesh.export(os.path.join(base_exp_dir, 'mesh_udf', 'mesh_res%d_thred%.4f_iter%d%s.ply'%(resolution, threshold, iteration, postfix)))
    
    def validate_mesh(self, resolution=64, threshold=0.0, base_exp_dir='', iteration=0, postfix=''):
        self.eval()
        if type(self.range_) == float:
            bound_min = torch.tensor([-self.range_,-self.range_,-self.range_]).cuda()
            bound_max = torch.tensor([self.range_,self.range_,self.range_]).cuda()
        else:
            bound_min = self.range_[0]
            bound_max = self.range_[1]
        print("saving")
        os.makedirs(os.path.join(base_exp_dir, 'mesh'), exist_ok=True)
        mesh = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: self.sdf(pts))
        mesh.export(os.path.join(base_exp_dir, 'mesh', 'mesh_res%d_thred%.4f_iter%d%s.ply'%(resolution, threshold, iteration, postfix)))
        self.train()

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh
    
    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
        max_val = -1
        min_val = 1000
        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        points = []
        colors = []
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        #print(pts.device)
                        
                        val = query_func(pts.cuda()).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        max_val = max(max_val, val.max())
                        min_val = min(min_val, val.min())
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        print(max_val, min_val)
        return u
    
    def sdf(self, x):
        return self.forward(x)



    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x, y=None):
        x.requires_grad_(True)
        if y is None:
            y = self.sdf(x)
        # y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
    
    def vis_field(self, base_exp_dir, point_num=100000, thred=0.2):
        if type(self.range_) == float:
            bound_min = torch.tensor([-self.range_,-self.range_,-self.range_]).cuda()
            bound_max = torch.tensor([self.range_,self.range_,self.range_]).cuda()
        else:
            bound_min = self.range_[0]
            bound_max = self.range_[1]

        X = torch.rand(point_num, 1).cuda()*(bound_max[0]- bound_min[0]) + bound_min[0]
        Y = torch.rand(point_num, 1).cuda()*(bound_max[1]- bound_min[1]) + bound_min[1]
        Z = torch.rand(point_num, 1).cuda()*(bound_max[2]- bound_min[2]) + bound_min[2]
        zeros = torch.zeros(point_num, 1).cuda()
        
        x_plane = torch.cat([zeros,Y,Z], dim=-1)
        y_plane = torch.cat([X, zeros, Z], dim=-1)
        z_plane = torch.cat([X, Y, zeros], dim=-1)
        x_val = []
        y_val = []
        z_val = []
        
        import matplotlib.cm as cm
        for i in range(0, point_num, 10000):
            with torch.no_grad():
                x_val_t = self.sdf(x_plane[i:i+10000]).cpu()
                y_val_t = self.sdf(y_plane[i:i+10000]).cpu()
                z_val_t = self.sdf(z_plane[i:i+10000]).cpu()
            x_val.append(x_val_t.numpy())
            y_val.append(y_val_t.numpy())
            z_val.append(z_val_t.numpy())
        x_val = np.concatenate(x_val).reshape(-1)
        y_val = np.concatenate(y_val).reshape(-1)
        z_val = np.concatenate(z_val).reshape(-1)
        if (x_val > thred).sum() > 0:
            x_val[x_val > thred] = thred
        if (y_val > thred).sum() > 0:
            y_val[y_val > thred] = thred
        if (z_val > thred).sum() > 0:
            z_val[z_val > thred] = thred

        x_val = (x_val - x_val.min())/(x_val.max()-x_val.min())
        y_val = (y_val - y_val.min())/(y_val.max()-y_val.min())
        z_val = (z_val - z_val.min())/(z_val.max()-z_val.min())
        color_x = cm.get_cmap('cool')(x_val)[..., :3]
        color_y = cm.get_cmap('cool')(y_val)[..., :3]
        color_z = cm.get_cmap('cool')(z_val)[..., :3]
        

        save_dir = os.path.join(base_exp_dir, 'outputs', 'field')
        os.makedirs(save_dir, exist_ok=True)
        print(x_plane.shape, color_x.shape)
        np.savetxt(os.path.join(save_dir, 'x_plane.txt'), np.concatenate((x_plane.cpu(), color_x), axis=1))
        np.savetxt(os.path.join(save_dir, 'y_plane.txt'), np.concatenate((y_plane.cpu(), color_y), axis=1))
        np.savetxt(os.path.join(save_dir, 'z_plane.txt'), np.concatenate((z_plane.cpu(), color_z), axis=1))
        

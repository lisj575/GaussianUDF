import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse

import trimesh
from pathlib import Path
import subprocess

import sys
sys.path.append("./")
sys.path.append("eval_dtu")
sys.path.append("..")
import render_utils as rend_util
from cameras import CamerasWrapper, load_gs_cameras
from tqdm import tqdm

def clean_mesh(mesh, source_path, gs_output_path, dataset_name='real360'):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh import rasterizer
    from pytorch3d.renderer.cameras import PerspectiveCameras
    # 加载相机
    cam_list = load_gs_cameras(
        source_path=source_path,
        gs_output_path=gs_output_path,
        load_gt_images=False,
    )
    training_cameras = CamerasWrapper(cam_list)
    image_height, image_width = training_cameras.gs_cameras[0].image_height, training_cameras.gs_cameras[0].image_width

    num_faces = len(mesh.faces)
    nb_visible = 3
    count = torch.zeros(num_faces, device="cuda")
    # K, R, t, sizes = cams[:4]

    n = len(training_cameras.gs_cameras)
    with torch.no_grad():
        for i in tqdm(range(n), desc="clean_faces"):
            vertices = torch.from_numpy(mesh.vertices).cuda().float()
            faces = torch.from_numpy(mesh.faces).cuda().long()
            meshes = Meshes(verts=[vertices],
                            faces=[faces])
            raster_settings = rasterizer.RasterizationSettings(image_size=(image_height, image_width),
                                                               faces_per_pixel=1)
            meshRasterizer = rasterizer.MeshRasterizer(training_cameras.p3d_cameras[i], raster_settings)

            with torch.no_grad():
                ret = meshRasterizer(meshes)
                pix_to_face = ret.pix_to_face
                # pix_to_face, zbuf, bar, pixd =

            visible_faces = pix_to_face.view(-1).unique()
            count[visible_faces[visible_faces > -1]] += 1

    pred_visible_mask = (count >= nb_visible).cpu()

    mesh.update_faces(pred_visible_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def cull_scan(scan, mesh_path, result_mesh_file):
    
    # load poses
    instance_dir = os.path.join('data/dtu', 'scan{0}'.format(scan))
    gs_path = os.path.dirname(os.path.dirname(os.path.dirname(result_mesh_file)))

    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    
    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path)
    mesh = clean_mesh(mesh, instance_dir, gs_path, 'dtu')
    # load transformation matrix

    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    for i in range(n_images):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]
            # print(f'culling {i}')
            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    
    # transform vertices to world 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)
    print(result_mesh_file)
    del mesh
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh')
    parser.add_argument('--output_dir', type=str, default='', help='path to the output folder')
    parser.add_argument('--DTU', type=str,  default='./data/DTU_GTpoints', help='path to the GT DTU point clouds')
    args = parser.parse_args()


    Offical_DTU_Dataset = args.DTU
    if args.output_dir == '':
        args.output_dir = os.path.dirname(args.input_mesh)
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(args.input_mesh):
        mesh_dir = os.listdir(args.input_mesh)
        mesh_dir = [os.path.join(args.input_mesh, f) for f in mesh_dir]
    else:
        mesh_dir = [args.input_mesh]
    print("all mesh:")
    print(mesh_dir)

    scan = args.scan_id
    for ply_file in mesh_dir:
        # if 'mesh_res' not in ply_file:
        #     continue
        if 'ply' not in ply_file or 'vis_' in ply_file:
            continue
        dir_name = ply_file[:-4]
        if os.path.exists(dir_name) and len(os.listdir(dir_name)) == 4:
            continue
        os.makedirs(dir_name, exist_ok=True)
        result_mesh_file = os.path.join(dir_name, "culled_mesh.ply")
        if not os.path.exists(result_mesh_file):
            cull_scan(scan, ply_file, result_mesh_file)

        cmd = f"python eval_dtu/eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {dir_name}"
        os.system(cmd)
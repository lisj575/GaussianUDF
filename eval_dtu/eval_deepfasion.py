import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse, os, sys

import cv2
import torch
sys.path.append("../")
sys.path.append("../../")
from pathlib import Path
import render_utils as rend_util
from cameras import CamerasWrapper, load_gs_cameras
from skimage.morphology import binary_dilation, disk
import trimesh
import torch.nn.functional as F

def get_path_components(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

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


def cull_scan(scan, mesh_path, result_mesh_file, thred=0):
    
    # load poses
    instance_dir = os.path.join('../data/deepfashion_rendering_fov60', str(scan))
    gs_path = os.path.dirname(os.path.dirname(result_mesh_file))
    if not os.path.exists(os.path.join(gs_path, 'cameras.json')):
        gs_path = os.path.join(f'../output/2dgs_lamb0_all/{scan}')
        if not os.path.exists(os.path.join(gs_path, 'cameras.json')):
            print("not exist!!!")
            exit()
    
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras_sphere.npz'.format(instance_dir)
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
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1024, 1024

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
            maski = torch.from_numpy(binary_dilation(maski, disk(2))).float()[None, None].cuda()


            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]
            # print(f'culling {i}')
            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    
    mask = (sampled_masks > thred).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    # triangle_clusters = np.asarray(triangle_clusters)
    # cluster_n_triangles = np.asarray(cluster_n_triangles)
    # cluster_area = np.asarray(cluster_area)
    # largest_cluster_idx = cluster_n_triangles.argmax()
    # triangles_to_remove = (triangle_clusters != largest_cluster_idx)
    
    # transform vertices to world 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    
    mesh.export(result_mesh_file)
    print(result_mesh_file)
    del mesh



if __name__ == '__main__':
    from glob import glob

    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--gt', type=str, help='ground truth')
    parser.add_argument('--scan', type=int, default=-1)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='../data/deepfashion_rendering_fov60')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.002)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=0.1)
    parser.add_argument('--visualize_threshold', type=float, default=0.01)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--target_name', type=str, default='')
    args = parser.parse_args()

    method = 'colmap'
    if args.scan == -1:
        scans = [30, 92, 117, 133, 164, 204, 300, 320, 448, 522, 591, 598]
    else:
        scans = [args.scan]

    use_cull = True
    avg_mean = []
    target_name = args.target_name
    for scan in scans:

        GT_DIR = f"{args.dataset_dir}/{scan}"
        base_dir = f"../output_df/{target_name}/{scan}/mesh_udf"
        filename = "mesh_res512_thred0.0000_iter30000.ply"

        if use_cull:
            mesh_path = base_dir
            if not os.path.exists(os.path.join(mesh_path, filename)):
                print("cull no such file :" + os.path.join(mesh_path, filename))
                continue
            name = 'culled_mesh1.ply'
            if not os.path.exists(os.path.join(mesh_path, name)):
                cull_scan(scan, os.path.join(mesh_path, filename), os.path.join(mesh_path, name), thred=0)
            base_dir = mesh_path
            filename = name

        args.data = os.path.join(base_dir, filename)

        if not os.path.exists(args.data):
            continue

        pparent, stem, ext = get_path_components(args.data)
        if args.log is None:
            path_log = os.path.join(pparent, 'eval_result%d.txt'%(int(use_cull)))
        else:
            path_log = args.log
        print("processing scan%d" % scan)
        if os.path.exists(path_log):
            with open(path_log, 'r') as f:
                first_line = f.readline().split(' ')
                cd = float(first_line[1])
                print(cd)
                avg_mean.append(cd)
            continue


        args.gt = os.path.join(GT_DIR, "%d_pc_swap.ply" % scan)
        args.vis_out_dir = os.path.join(base_dir, "scan{}".format(scan))
        args.scan = scan
        os.makedirs(args.vis_out_dir, exist_ok=True)

        dist_thred1 = 0.001
        dist_thred2 = 0.002

        thresh = args.downsample_density

        if args.mode == 'mesh':
            pbar = tqdm(total=9)
            pbar.set_description('read data mesh')
            data_mesh = o3d.io.read_triangle_mesh(args.data)

            vertices = np.asarray(data_mesh.vertices)
            triangles = np.asarray(data_mesh.triangles)
            tri_vert = vertices[triangles]

            pbar.update(1)
            pbar.set_description('sample pcd from mesh')
            v1 = tri_vert[:, 1] - tri_vert[:, 0]
            v2 = tri_vert[:, 2] - tri_vert[:, 0]
            l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
            l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
            area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
            non_zero_area = (area2 > 0)[:, 0]
            l1, l2, area2, v1, v2, tri_vert = [
                arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
            ]
            thr = thresh * np.sqrt(l1 * l2 / area2)
            n1 = np.floor(l1 / thr)
            n2 = np.floor(l2 / thr)

            with mp.Pool() as mp_pool:
                new_pts = mp_pool.map(sample_single_tri,
                                      ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                                       range(len(n1))), chunksize=1024)

            new_pts = np.concatenate(new_pts, axis=0)
            data_pcd = np.concatenate([vertices, new_pts], axis=0)

        elif args.mode == 'pcd':
            pbar = tqdm(total=8)
            pbar.set_description('read data pcd')
            data_pcd_o3d = o3d.io.read_point_cloud(args.data)
            data_pcd = np.asarray(data_pcd_o3d.points)

        pbar.update(1)
        pbar.set_description('random shuffle pcd index')
        shuffle_rng = np.random.default_rng()
        shuffle_rng.shuffle(data_pcd, axis=0)

        pbar.update(1)
        # pbar.set_description('downsample pcd')
        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(data_pcd)
        rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
        mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
        for curr, idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[idxs] = 0
                mask[curr] = 1
        data_down = data_pcd[mask]
        # data_down = data_pcd

        pbar.update(1)
        pbar.set_description('read STL pcd')
        stl_pcd = o3d.io.read_point_cloud(args.gt)
        stl = np.asarray(stl_pcd.points)

        pbar.update(1)
        pbar.set_description('compute data2stl')
        nn_engine.fit(stl)
        dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)
        max_dist = args.max_dist
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

        precision_1 = len(dist_d2s[dist_d2s < dist_thred1]) / len(dist_d2s)
        precision_2 = len(dist_d2s[dist_d2s < dist_thred2]) / len(dist_d2s)

        pbar.update(1)
        pbar.set_description('compute stl2data')

        nn_engine.fit(data_down)
        dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

        recall_1 = len(dist_s2d[dist_s2d < dist_thred1]) / len(dist_s2d)
        recall_2 = len(dist_s2d[dist_s2d < dist_thred2]) / len(dist_s2d)

        pbar.update(1)
        pbar.set_description('visualize error')
        vis_dist = args.visualize_threshold
        R = np.array([[1, 0, 0]], dtype=np.float64)
        G = np.array([[0, 1, 0]], dtype=np.float64)
        B = np.array([[0, 0, 1]], dtype=np.float64)
        W = np.array([[1, 1, 1]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color = R * data_alpha + W * (1 - data_alpha)
        data_color[dist_d2s[:, 0] >= max_dist] = G
        write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2gt.ply', data_down, data_color)
        stl_color = np.tile(B, (stl.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color = R * stl_alpha + W * (1 - stl_alpha)
        stl_color[dist_s2d[:, 0] >= max_dist] = G
        write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_gt2d.ply', stl, stl_color)

        pbar.update(1)
        pbar.set_description('done')
        pbar.close()
        over_all = (mean_d2s + mean_s2d) / 2

        fscore_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-6)
        fscore_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-6)

        print(f'over_all: {over_all}; mean_d2gt: {mean_d2s}; mean_gt2d: {mean_s2d}.')
        print(f'precision_1mm: {precision_1};  recall_1mm: {recall_1};  fscore_1mm: {fscore_1}')
        print(f'precision_2mm: {precision_2};  recall_2mm: {recall_2};  fscore_2mm: {fscore_2}')

        avg_mean.append(np.round(over_all, 6))
        with open(path_log, 'w+') as fLog:
            fLog.write(f'over_all {np.round(over_all, 6)} '
                       f'mean_d2gt {np.round(mean_d2s, 6)} '
                       f'mean_gt2d {np.round(mean_s2d, 6)} \n'
                       f'precision_1mm {np.round(precision_1, 6)} '
                       f'recall_1mm {np.round(recall_1, 6)} '
                       f'fscore_1mm {np.round(fscore_1, 6)} \n'
                       f'precision_2mm {np.round(precision_2, 6)} '
                       f'recall_2mm {np.round(recall_2, 6)} '
                       f'fscore_2mm {np.round(fscore_2, 6)} \n'
                       f'[{stem}] \n')
    

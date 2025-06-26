from warp_func import depth_to_world_pts
import numpy as np
from tqdm import tqdm
import sklearn.neighbors as skln
from scipy.io import loadmat
import open3d as o3d
import multiprocessing as mp
import os

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def color_pts_error(data_pcd, scan_id, vis_out_dir):
    thresh = 0.2
    pbar = tqdm(total=8)
    pbar.set_description('read data pcd')
    print("pts shape:", data_pcd.shape)
    #data_pcd = data_pcd[np.random.choice(data_pcd.shape[0], 300000)]
    
    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)
    print(data_pcd.shape)
    pbar.update(1)
    pbar.set_description('downsample pcd') # just remove the repeated points
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]
    

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'data/DTU_GTpoints/ObsMask/ObsMask{scan_id}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = 60

    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3 # xyz in bounding box; less than min, bigger than max
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]
    
    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'data/DTU_GTpoints/Points/stl/stl{scan_id:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)

    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = 20
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'data/DTU_GTpoints/ObsMask/Plane{scan_id}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = 10
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
    data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{vis_out_dir}_vis_{scan_id:03}_d2s.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
    stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{vis_out_dir}_vis_{scan_id:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)
    
    import json
    with open(f'{vis_out_dir}_results.json', 'w') as fp:
        json.dump({
            'mean_d2s': mean_d2s,
            'mean_s2d': mean_s2d,
            'overall': over_all,
        }, fp, indent=True)

def color_depth_error(depth, view_cam, scan_id, out_dir='tmp_results/0911-color_depth'):
    
    pts = depth_to_world_pts(depth, view_cam.K, view_cam.w2c)

    cam_file = 'data/dtu/scan%d/cameras.npz'%scan_id
    camera_dict = np.load(cam_file)
    scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
    scaled_pts = pts.cpu() * scale_mat[0,0] + scale_mat[:3, 3][None]
    os.makedirs(out_dir, exist_ok=True)
    color_pts_error(scaled_pts, scan_id, out_dir + '/' + view_cam.img_name.split('.')[0])

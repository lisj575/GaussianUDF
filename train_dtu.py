#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
import random
import torch.backends
import torch.backends.cudnn
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from np_field import UDFNetwork
from pytorch3d.ops import knn_points
import torch.nn.functional as F
import time
import numpy as np
from submodules.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from math import sqrt

def seed_all(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

time_flag = True
now_time = 0
def time_print(info=''):
    global time_flag, now_time
    time_flag = ~time_flag
    new_time = time.time()
    if time_flag:
        print(info, (new_time - now_time) * 1000)
    now_time = new_time

def patch_homography(H, uv):
    # H: [batch_size, nsrc, 3, 3]
    # uv: [batch_size, 121, 2]
    N, Npx = uv.shape[:2]
    H = H.permute(1, 0, 2, 3)
    Nsrc = H.shape[0]
    H = H.view(Nsrc, N, -1, 3, 3)
    ones = torch.ones(uv.shape[:-1], device=uv.device).unsqueeze(-1)
    hom_uv = torch.cat((uv, ones), dim=-1)

    tmp = torch.einsum("vprik,pok->vproi", H, hom_uv)
    tmp = tmp.reshape(Nsrc, -1, 3)

    grid = tmp[..., :2] / (tmp[..., 2:] + 1e-8)

    return grid


def depth_norm_to_world(depth, normal, intrinsic, extrinsic, pts_num=-1):
    _, image_height, image_width = normal.shape
    if pts_num == -1:
        pts_num_w = image_width
        pts_num_h = image_height
        pts_num = pts_num_w * pts_num_h
    else:
        pts_num_w = pts_num_h = pts_num

    pixels_x = torch.randint(low=0, high=image_width, size=[pts_num_w], device=normal.device)
    pixels_y = torch.randint(low=0, high=image_height, size=[pts_num_h], device=normal.device)
    sub_normal = normal.permute(1,2,0)[(pixels_y, pixels_x)]    # batch_size, 3
    sub_depth = depth.permute(1,2,0)[(pixels_y, pixels_x)]
    pixel = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) * sub_depth
    sub_normal = sub_normal.transpose(0,1)
    pixel = pixel.transpose(0,1)

    inv_intrinsic = torch.inverse(intrinsic)
    cam_normal = torch.matmul(inv_intrinsic.float(), sub_normal.float())
    cam_pts = torch.matmul(inv_intrinsic.float(), pixel.float())
    ones = torch.ones([1, pts_num], device=depth.device)
    cam_normal_hom = torch.cat([cam_normal, ones], dim=0)
    cam_pts_hom = torch.cat([cam_pts, ones], dim=0)
    c2w = torch.inverse(extrinsic)
    world_pts = torch.matmul(c2w, cam_pts_hom).transpose(1,0)[:,:3]
    world_normal = torch.matmul(c2w, cam_normal_hom).transpose(1,0)[:,:3]
    return world_pts, world_normal

def compute_LNCC(ref_gray, src_grays):
    # ref_gray: [1, batch_size, 121, 1]
    # src_grays: [nsrc, batch_size, 121, 1]
    ref_gray = ref_gray.permute(1, 0, 3, 2)  # [batch_size, 1, 1, 121]
    src_grays = src_grays.permute(1, 0, 3, 2)  # [batch_size, nsrc, 1, 121]

    ref_src = ref_gray * src_grays  # [batch_size, nsrc, 1, npatch]

    bs, nsrc, nc, npatch = src_grays.shape
    patch_size = int(sqrt(npatch))
    ref_gray = ref_gray.view(bs, 1, 1, patch_size, patch_size).view(-1, 1, patch_size, patch_size)
    src_grays = src_grays.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)
    ref_src = ref_src.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)

    ref_sq = ref_gray.pow(2)
    src_sq = src_grays.pow(2)

    filters = torch.ones(1, 1, patch_size, patch_size, device=ref_gray.device)
    padding = patch_size // 2

    ref_sum = F.conv2d(ref_gray, filters, stride=1, padding=padding)[:, :, padding, padding]
    src_sum = F.conv2d(src_grays, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
    ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
    src_sq_sum = F.conv2d(src_sq, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
    ref_src_sum = F.conv2d(ref_src, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)

    u_ref = ref_sum / npatch
    u_src = src_sum / npatch

    cross = ref_src_sum - u_src * ref_sum - u_ref * src_sum + u_ref * u_src * npatch
    ref_var = ref_sq_sum - 2 * u_ref * ref_sum + u_ref * u_ref * npatch
    src_var = src_sq_sum - 2 * u_src * src_sum + u_src * u_src * npatch

    cc = cross * cross / (ref_var * src_var + 1e-5)  # [batch_size, nsrc, 1, npatch]
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc, _ = torch.topk(ncc, 4, dim=1, largest=False)
    ncc = torch.mean(ncc, dim=1, keepdim=True)

    return ncc



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    
    first_iter = 0
    pc_bbx = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]]).cuda()

    tb_writer, tb_writer2 = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    field = UDFNetwork(range_=pc_bbx).cuda()
    gaussians.field = field
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    inner_iter = 1
    iteration = 1
    chamfer =  ChamferDistanceL1().cuda()

    if checkpoint:
        print("loading from %s" % checkpoint)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        iteration = first_iter
        gaussians.field.validate_mesh(resolution=128, threshold=0.005, base_exp_dir=scene.model_path, iteration=iteration)
        gaussians.field.vis_field(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_stack_raw = scene.getTrainCameras().copy()
    
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0


    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # params of far loss
    start_far_iteration = 9000
    far1_num = 2000
    far2_num = 5000
    far1_std = -1#0.25
    far2_std = 0.02
    far_weight = 1.0

    # params of projection loss
    use_projection_loss = True
    start_projection_iteration = 12000
    projection_weight = 0.15


    # params of near loss 
    use_near_loss =  True
    start_near_iteration = 12000
    near_gs_center = 500
    root_num = 10
    near_weight = 1.0
    T = 0.02
    use_warp_loss = True
    warp_num = 2000

    record_path = os.path.join(scene.model_path, 'record')
    os.makedirs(record_path, exist_ok=True)
    os.system('cp *.py %s'%record_path)
    os.system('cp scene/gaussian_model.py %s'%record_path)

    if iteration > start_far_iteration:
        inner_iter = (iteration - start_far_iteration) * 5

    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    for batch in range(9_999_999):
        if iteration >= opt.iterations + 1:
            break
            
        if iteration <= start_far_iteration:
            rgb_flag = True
            udf_flag = False
        else:
            udf_flag = True
            rgb_flag = False
            if inner_iter % 5 == 0:
                rgb_flag = True

        udf_loss = 0
        if udf_flag:
            far_loss = 0
            projection_loss = 0
            near_loss = 0
            ncc_loss = 0.0
            gaussians.update_learning_rate_np(inner_iter)
            origin_gs_pts = gaussians.get_xyz.clone()
            gs_pts = origin_gs_pts.clone()
            mask_x = (gs_pts[:, 0] < pc_bbx[1,0]) & (gs_pts[:,0] > pc_bbx[0,0])
            mask_y = (gs_pts[:, 1] < pc_bbx[1,1]) & (gs_pts[:,1] > pc_bbx[0,1])
            mask_z = (gs_pts[:, 2] < pc_bbx[1,2]) & (gs_pts[:,2] > pc_bbx[0,2])
            mask = mask_x & mask_y & mask_z
            gs_pts = gs_pts[mask]

            scaling = gaussians.get_scaling[mask].detach()
            scaling_thred = scaling.max(dim=1)[0].mean() * 3
            origin_index = torch.arange(origin_gs_pts.shape[0], device=gs_pts.device)[mask]
            total_num = gs_pts.shape[0]
            
            # #==============far loss=================

            if far1_num > 0:
                if far1_std > 0:
                    sub_pts_index = torch.randint(total_num, [far1_num]).cuda()
                    far1_samp = gs_pts[sub_pts_index] + torch.randn(sub_pts_index.shape[0], 3, device=gs_pts.device) * far1_std
                else:
                    far1_samp = torch.rand(far1_num, 3, device=gs_pts.device) * (pc_bbx[1:,:] - pc_bbx[0:1,:]) + pc_bbx[0:1,:]

            sub_pts_index = torch.randint(total_num, [far2_num]).cuda()
            far2_samp = gs_pts[sub_pts_index] + torch.randn(sub_pts_index.shape[0], 3, device=gs_pts.device) * far2_std
            
            if far1_num > 0:
                far_samples = torch.cat([far2_samp, far1_samp], 0).detach()
            else:
                far_samples = far2_samp

            index = torch.randperm(far_samples.shape[0], device=gs_pts.device)
            far_samples = far_samples[index]

            with torch.no_grad():
                knns = knn_points(far_samples[None], gs_pts[None], K=1)
                knn_idx = knns.idx[0][:, 0]
            surf_points = gs_pts[knn_idx]
            
            udf = gaussians.field.sdf(far_samples)
            positions_grad = gaussians.field.gradient(far_samples).squeeze(1)
            grad_norm = F.normalize(positions_grad, dim=1)
            samples_moved = far_samples - grad_norm * udf

            pull_distance = chamfer(samples_moved.unsqueeze(0), surf_points.detach().unsqueeze(0))
            far_loss = pull_distance.mean()

            udf_loss = udf_loss + far_loss * far_weight
            
            #==============projection loss=================
            if use_projection_loss and iteration >= start_projection_iteration:
                batch_gs_pts = surf_points
                batch_gs_udf = gaussians.field.sdf(batch_gs_pts)
                batch_gs_gradient = gaussians.field.gradient(batch_gs_pts).squeeze(1)
                batch_gs_gradient = torch.nn.functional.normalize(batch_gs_gradient, dim=-1)
                surf_points_moved = batch_gs_pts - batch_gs_udf * batch_gs_gradient
                projection_loss = torch.linalg.norm((surf_points_moved.detach() - batch_gs_pts), ord=2, dim=-1).mean()
                udf_loss = udf_loss + projection_weight * projection_loss

            #==============near loss=================
            if use_near_loss and iteration >= start_near_iteration:

                surf_ind = torch.randint(gs_pts.shape[0], [near_gs_center], device=gs_pts.device)

                surf_scaling = scaling[surf_ind]
                big_scale_mask = surf_scaling.max(dim=1)[0]  > scaling_thred
                if big_scale_mask.sum() > 0:
                    new_surf_ind = torch.cat([surf_ind, surf_ind[big_scale_mask]])
                    index = torch.randperm(new_surf_ind.shape[0], device=gs_pts.device)
                    surf_ind = new_surf_ind[index]

                origin_surf_ind = origin_index[surf_ind]
                new_plane_center = origin_surf_ind.shape[0]
                
                trans = gaussians.get_covariance(1.0, origin_surf_ind).transpose(1,2)
                gs_normals = trans[:,:3,2].clone()
                trans[:,:,2] = 0 

                root_points = torch.ones(new_plane_center * root_num, 4, device=gs_pts.device)
                root_points[:,:3] = torch.randn(new_plane_center * root_num, 3, device=gs_pts.device)
                root_points = root_points.reshape(new_plane_center, root_num, 4)
                root_pts = torch.matmul(trans, root_points.transpose(1,2)).transpose(1,2)[...,:3]
                moved_normals = gs_normals
                surf_pts_sample = root_pts.reshape(-1,3)

                #### self-supervision
                gradient = moved_normals.unsqueeze(1).repeat(1,root_num,1).reshape(-1,3)
                delta1 = torch.rand(surf_pts_sample.shape[0], 1, device=gs_pts.device) * T
                delta2 = torch.rand(surf_pts_sample.shape[0], 1, device=gs_pts.device).cuda() * T
                samples_pos = (surf_pts_sample + gradient * delta1)
                samples_neg = (surf_pts_sample - gradient * delta2)

                samples = torch.cat((samples_pos, samples_neg), dim=0)
                delta = torch.cat((delta1,delta2),dim=0)
                index = torch.randperm(samples.shape[0], device=gs_pts.device)
                samples = samples[index]
                delta = delta[index]
                udf = gaussians.field.sdf(samples)
                near_loss = abs(udf-delta).mean()

                udf_loss = udf_loss + near_loss * near_weight

            inner_iter += 1
            if tb_writer2 is not None:
                tb_writer2.add_scalar('field_loss/udf_total_loss', udf_loss, inner_iter)
                tb_writer2.add_scalar('field_loss/far_loss', far_loss, inner_iter)
                if use_projection_loss:
                    tb_writer2.add_scalar('field_loss/projection_loss', projection_loss, inner_iter)
                if use_near_loss:
                    tb_writer2.add_scalar('field_loss/near_loss', near_loss, inner_iter)
                if use_warp_loss:
                    tb_writer2.add_scalar('field_loss/warp_loss', ncc_loss, inner_iter)
                
        
        rgb_total_loss = 0
        if rgb_flag:
            iteration += 1
            iter_start.record()
            
            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                
                viewpoint_stack = scene.getTrainCameras().copy()
    
            choose_ind = randint(0, len(viewpoint_stack)-1)
            viewpoint_cam = viewpoint_stack.pop(choose_ind)
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda()
        
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # regularization
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
            
            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            if use_warp_loss and iteration > 7000:
               
                img_id = viewpoint_cam.image_id
                #render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                depth = render_pkg['surf_depth']
                normal = render_pkg['rend_normal']
                pts_render, normal_render = depth_norm_to_world(depth, normal, viewpoint_cam.K, viewpoint_cam.w2c, warp_num)
                pts_sdf0 = pts_render
                gradients_sdf0 = normal_render
                    
                    
                if len(pts_sdf0) > 0:
                    
                    gradients_sdf0 = gradients_sdf0.reshape(-1, 1, 3)
                    gradients_sdf0 = gradients_sdf0 / (1e-5 + torch.linalg.norm(gradients_sdf0, ord=2, dim=-1, keepdim=True))
                    
                    pts_sdf0 = pts_sdf0.reshape(-1,1,3)
                    batch_size = gradients_sdf0.shape[0]
                    
                    gradients_sdf0 = torch.matmul(scene.pose_all[img_id, :3, :3].permute(1, 0)[None, ...], gradients_sdf0.permute(0, 2, 1)).permute(0, 2, 1).detach()
                    project_xyz = torch.matmul(scene.pose_all[img_id, :3, :3].permute(1, 0), pts_sdf0.permute(0, 2, 1))
                    t = - torch.matmul(scene.pose_all[img_id, :3, :3].permute(1, 0), scene.pose_all[img_id, :3, 3, None])
                    project_xyz = project_xyz + t
                    
                    pts_sdf0_ref = project_xyz
                    
                    project_xyz = torch.matmul(scene.intrinsics_all[img_id, :3, :3], project_xyz)  # [batch_size, 3, 1]
                    mid_inside_sphere = (pts_sdf0.reshape(-1,3) ** 2).sum(dim=1, keepdim=True).float().detach() < 1
                    disp_sdf0 = torch.matmul(gradients_sdf0, pts_sdf0_ref)
                    
                    # Compute Homography
                    K_ref_inv = scene.intrinsics_all_inv[img_id, :3, :3]
                    src_idx = scene.src_idx[img_id][:9]
                    intrinsics = scene.intrinsics_all[src_idx]
                    K_src = intrinsics[:, :3, :3]
                    num_src = K_src.shape[0]
                    R_ref_inv = scene.pose_all[img_id, :3, :3]
                    poses = scene.pose_all[src_idx]
                    R_src = poses[:, :3, :3].permute(0, 2, 1)
                    C_ref = scene.pose_all[img_id, :3, 3]
                    C_src = poses[:, :3, 3]
                    R_relative = torch.matmul(R_src, R_ref_inv)
                    C_relative = C_ref[None, ...] - C_src
                    tmp = torch.matmul(R_src, C_relative[..., None])
                    
                    tmp = torch.matmul(tmp[None, ...].expand(batch_size, num_src, 3, 1), gradients_sdf0.expand(batch_size, num_src, 3)[..., None].permute(0, 1, 3, 2))  # [Batch_size, num_src, 3, 1]

                    tmp = R_relative[None, ...].expand(batch_size, num_src, 3, 3) + tmp / (disp_sdf0[..., None] + 1e-5)
                    tmp = torch.matmul(K_src[None, ...].expand(batch_size, num_src, 3, 3), tmp)
                    
                    Hom = torch.matmul(tmp, K_ref_inv[None, None, ...])
                    
                    pixels_x = project_xyz[:, 0, 0] / (project_xyz[:, 2, 0] + 1e-5)
                    pixels_y = project_xyz[:, 1, 0] / (project_xyz[:, 2, 0] + 1e-5)
                    pixels = torch.stack([pixels_x, pixels_y], dim=-1).float()
                    h_patch_size = 5
                    total_size = (h_patch_size * 2 + 1) ** 2
                    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=pts_sdf0.device)
                    offsets = torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)

                    pixels_patch = pixels.view(batch_size, 1, 2) + offsets.float()  # [batch_size, 121, 2]
                    ref_image = scene.images_gray[img_id, :, :]
                    src_images = scene.images_gray[src_idx, :, :]
                    
                    h, w = ref_image.shape
                    grid = patch_homography(Hom, pixels_patch)
                    
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (w - 1) - 1.0
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (h - 1) - 1.0
                    sampled_gray_val = F.grid_sample(src_images.unsqueeze(1), grid.view(num_src, -1, 1, 2), align_corners=True)
                    
                    sampled_gray_val = sampled_gray_val.view(num_src, batch_size, total_size, 1)  # [nsrc, batch_size, 121, 1]
                    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (w - 1) - 1.0
                    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (h - 1) - 1.0
                    grid = pixels_patch.detach()
                    
                    ref_gray_val = F.grid_sample(ref_image[None, None, ...], grid.view(1, -1, 1, 2), align_corners=True)
                    ref_gray_val = ref_gray_val.view(1, batch_size, total_size, 1)

                    ncc = compute_LNCC(ref_gray_val, sampled_gray_val)
                    ncc = ncc * mid_inside_sphere
                    ncc_loss = ncc.mean() * 0.5
            else:
                ncc_loss = 0.0

            # loss
            rgb_total_loss = loss + dist_loss + normal_loss + ncc_loss
            
        total_loss = udf_loss + rgb_total_loss
        total_loss.backward()

        if rgb_flag:
            iter_end.record()
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "Points": f"{len(gaussians.get_xyz)}",
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                if tb_writer is not None:
                    tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                        
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
            if iteration % 5000 == 0 and iteration >= start_far_iteration:
                with torch.no_grad():
                    gaussians.field.validate_mesh(resolution=256, threshold=0.005, base_exp_dir=scene.model_path, iteration=iteration)
                    torch.save((gaussians.field.state_dict(), iteration), scene.model_path + "/field" + str(iteration) + ".pth")
                        
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)
    gaussians.field.nudf_extract(resolution=512, base_exp_dir=scene.model_path, dist_threshold_ratio=2.0, threshold=0.0, iteration=opt.iterations, voxel_origin=pc_bbx[0].cpu().numpy().tolist())

        
def prepare_output_and_logger(args):   
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join(args.outdir, unique_str[0:10])
    else:
        if args.model_path != 'debug':
            current_date = time.strftime("%Y%m%d", time.localtime())
            args.model_path = args.outdir+"/"+current_date+'-'+args.model_path
        else:
            args.model_path = args.outdir+"/debug"

    # Set up output folder
    
    print("Output folder: {}".format(os.path.join(args.model_path)))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    tb_writer2 = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)

        path = os.path.join(args.model_path, 'field')
        os.makedirs(path, exist_ok=True)
        tb_writer2 = SummaryWriter(path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, tb_writer2

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 9000, 12000, 20000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 9000, 12000, 20000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1, 7000, 9000,12000, 15000, 20000, 25000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=int, default=0)
    #parser.add_argument("--outdir", type=str, default='output_v1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Optimizing " + args.model_path)
    seed_all()
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
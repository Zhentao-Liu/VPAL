import torch.nn.functional as F
import torch
import torch.nn as nn
from models.model import *
import math
import numpy as np

def angle2vec(PrimaryAngle, SecondaryAngle, isocenter, sid, sad, proj_spacing_x, proj_spacing_y):
    # input: PrimaryAngle, SecondaryAngle in rad
    # output: vec [12]
    # for flat panel detector

    cam_x = isocenter[0] + sad * np.cos(SecondaryAngle) * np.cos(PrimaryAngle)
    cam_y = isocenter[1] + sad * np.cos(SecondaryAngle) * np.sin(PrimaryAngle)
    cam_z = isocenter[2] + sad * np.sin(SecondaryAngle)
    cam = np.array([cam_x, cam_y, cam_z])

    det_x = isocenter[0] - (sid-sad) * np.cos(SecondaryAngle) * np.cos(PrimaryAngle)
    det_y = isocenter[1] - (sid-sad) * np.cos(SecondaryAngle) * np.sin(PrimaryAngle)
    det_z = isocenter[2] - (sid-sad) * np.sin(SecondaryAngle)
    det = np.array([det_x, det_y, det_z])

    u_x = -proj_spacing_x * np.sin(PrimaryAngle)
    u_y = proj_spacing_x  * np.cos(PrimaryAngle)
    u_z = 0
    
    v_x = proj_spacing_y * np.sin(SecondaryAngle) * np.cos(PrimaryAngle)
    v_y = proj_spacing_y * np.sin(SecondaryAngle) * np.sin(PrimaryAngle)
    v_z = -proj_spacing_y * np.cos(SecondaryAngle)   # z轴是负的

    u_vector = np.array([u_x, u_y, u_z])
    v_vector = np.array([v_x, v_y, v_z])
    vec = np.concatenate([cam, det, u_vector, v_vector])
    return vec

def get_rays(poses, H, W, fx, fy, deltaH=-1, deltaW=-1):
    '''
    :param poses: (N, 4, 4)
    :return rays: (N, W, H, 6)
    '''
    device = poses.device
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32) - (H+deltaH) * 0.5,
                          torch.arange(W, dtype=torch.float32) - (W+deltaW) * 0.5, )
    x = x.to(device)/fx
    y = y.to(device)/fy
    dirs = torch.stack([x, y, torch.ones_like(x)], -1,).to(torch.float32).to(device)
    rot = poses[:, :3, :3].transpose(1,2)
    trans = -torch.bmm(rot, poses[:,:3, 3:])
    cam_centers = trans.view(poses.shape[0],1,1,3).expand(poses.shape[0],H,W,3)
    cam_raydir = torch.matmul(rot[:,None,None,:,:], dirs.unsqueeze(-1)).squeeze(-1)
    cam_raydir = cam_raydir / torch.norm(cam_raydir,dim=-1,keepdim=True)
    rays = torch.concat([cam_centers, cam_raydir], dim=-1)
    return rays

def get_rays_projection(P, H, W):
    '''
    :param P: (N, 3, 4)  # projection matrix
    :return rays: (N, W, H, 6)
    '''
    device = P.device
    P33 = P[:, :, :3]  # [N, 3, 3]
    P4 = P[:, :, 3].unsqueeze(-1)  # [N, 3, 1]
    P33_inv = torch.inverse(P33) 
    cam_centers = -torch.bmm(P33_inv, P4)[..., 0] # [N, 3]
    cam_centers = cam_centers.view(P.shape[0], 1, 1, 3).expand(P.shape[0], H, W, 3) # [N, H, W, 3]

    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                          torch.arange(W, dtype=torch.float32, device=device))
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], -1,).to(torch.float32).to(device).reshape(-1, 3)  # [H*W, 3]
    pixel_coords = pixel_coords.view(1, H*W, 3, 1).expand(P.shape[0], H*W, 3, 1) # [N, H*W, 3, 1]
    P4 = P4.view(P.shape[0], 1, 3, 1).expand(P.shape[0], H*W, 3, 1) # [N, H*W, 3, 1] 

    depth = 1
    world_coords = torch.matmul(P33_inv[:, None, :, :], pixel_coords*depth - P4)[..., 0]  # [N,  H*W, 3]
    world_coords = world_coords.view(P.shape[0], H, W, 3)  # [N, H, W, 3]
    cam_raydir = world_coords - cam_centers  # [N, H, W, 3]
    cam_raydir = cam_raydir / torch.norm(cam_raydir,dim=-1,keepdim=True) # [N, H, W, 3]
    rays = torch.concat([cam_centers, cam_raydir], dim=-1)  # [N, H, W, 6]
    return rays

def get_pixel00_center(detectors, uvectors, vvectors, H, W):
    ## detectors, uvectors, vectors, [N, 3]
    ## pixel00_center, [N, 3]
    device = detectors.device
    float_H = torch.tensor(H).to(device)
    float_W = torch.tensor(W).to(device)
    v_offset = torch.floor(float_H/2) + torch.floor((float_H+1)/2) - (float_H+1)/2
    u_offset = torch.floor(float_W/2) + torch.floor((float_W+1)/2) - (float_W+1)/2
    pixel00_center = detectors - u_offset * uvectors - v_offset * vvectors
    return pixel00_center

def get_rays_vec(vecs, H, W):
    '''
    :param vecs: [N, 12]
    :return rays: [N, H, W, 6]
    '''
    ## for flat panel detector
    device = vecs.device
    N = vecs.shape[0]
    sources, detectors, uvectors, vvectors = vecs[:, :3], vecs[:, 3:6], vecs[:, 6:9], vecs[:, 9:]    # (N, 3)

    pixel00_center = get_pixel00_center(detectors, uvectors, vvectors, H, W)   # (N, 3)
    row_indices, col_indices = torch.meshgrid(torch.arange(H, device=device),    # [0, H - 1], [0, W - 1]
                                              torch.arange(W, device=device),
                                              indexing='ij')
    row_indices = row_indices.expand(N, -1, -1).unsqueeze(-1)
    col_indices = col_indices.expand(N, -1, -1).unsqueeze(-1)
    pix_coords = pixel00_center[:, None, None, :] + col_indices * uvectors[:, None, None, :] + row_indices * vvectors[:, None, None, :]

    rays_origin = sources.view(N, 1, 1, 3).expand(-1, H, W, -1)
    rays_dirs = pix_coords - rays_origin
    rays_dirs = rays_dirs / torch.linalg.norm(rays_dirs, dim=-1, keepdim=True)
    rays = torch.cat((rays_origin, rays_dirs), dim=3) 
    return rays

def sample_volume(rays_origins, rays_dirs, volume_origin, volume_phy, render_step_size,):
    # sample interval (flatten)

    device = rays_origins.device
    near, far = ray_AABB(rays_origins, rays_dirs, volume_origin, volume_phy)

    dis = far - near

    _, index = torch.sort(dis, dim=0)
    near_ = near[index[-1]]
    far_ = far[index[-1]]
    max_dis = far_ - near_

    N_sample = math.ceil(max_dis / render_step_size)

    t_step_start = torch.linspace(0, 1-1/N_sample, N_sample, device=device)
    t_step_end = torch.linspace(1/N_sample, 1, N_sample, device=device)
    
    t_start = near + max_dis * t_step_start
    t_end = near + max_dis * t_step_end
    mask = outer_mask((t_start + t_end)/2, near, far)
    t_start = t_start[mask==1]
    t_end = t_end[mask==1]
    ray_indices = torch.where(mask==1)[0]
    return ray_indices, t_start, t_end

def ray_AABB(rays_origins, rays_dirs, volume_origin, volume_phy,):
    device = rays_origins.device
    xyz_max = volume_phy + volume_origin
    xyz_min = volume_origin
    
    eps = torch.tensor(1e-6, dtype=torch.float32, device=device)
    vx = torch.where(rays_dirs[:,0]==0, eps, rays_dirs[:,0])
    vy = torch.where(rays_dirs[:,1]==0, eps, rays_dirs[:,1])
    vz = torch.where(rays_dirs[:,2]==0, eps, rays_dirs[:,2])
    
    ax = (xyz_max[0] - rays_origins[:,0]) / vx
    ay = (xyz_max[1] - rays_origins[:,1]) / vy
    az = (xyz_max[2] - rays_origins[:,2]) / vz
    bx = (xyz_min[0] - rays_origins[:,0]) / vx
    by = (xyz_min[1] - rays_origins[:,1]) / vy
    bz = (xyz_min[2] - rays_origins[:,2]) / vz

    t_min = torch.max(torch.max(torch.min(ax, bx), torch.min(ay, by)), torch.min(az, bz))
    t_max = torch.min(torch.min(torch.max(ax, bx), torch.max(ay, by)), torch.max(az, bz))
    
    return t_min.unsqueeze(1), t_max.unsqueeze(1)

def if_intersect(rays_origins, rays_dirs, volume_origin, volume_phy):
    device = rays_origins.device
    near, far = ray_AABB(rays_origins, rays_dirs, volume_origin, volume_phy)
    dis = far - near
    _, index = torch.sort(dis, dim=0)
    near_ = near[index[-1]]
    far_ = far[index[-1]]
    max_dis = far_ - near_
    if max_dis <= 0:
        return 0
    else:
        return 1

def outer_mask(z_samp,near,far):
    zero_mask_1 = (z_samp>=near)
    zero_mask_2 = (z_samp<=far)
    zero_mask = zero_mask_1 * zero_mask_2 + 0
    return zero_mask

def mu2ct(pix, mu_water=0.022,):
    ct = (pix/mu_water-1)*1000
    return ct

def ct2mu(pix, mu_water=0.022,):
    mu = (pix/1000+1)*mu_water
    return mu

def volume_sampling(xyz, volume, transfer=True):
    volume = volume.unsqueeze(0).unsqueeze(0).to(torch.float32)
    xyz = xyz.unsqueeze(0)
    xyz = xyz.unsqueeze(2)
    xyz = xyz.unsqueeze(2)
    samples = F.grid_sample(
        volume,
        xyz,
        align_corners=True,
        padding_mode="zeros",
    )
    if transfer:
        mu = ct2mu(samples[0,0,...])
    else:
        mu = samples[0,0,...]
    return mu[..., 0]

def composite(rays, model, occgrid, transfer, sample_mode, chunksize, volume_origin, volume_phy, render_step_size):
    '''
      rendering function
    '''
    if sample_mode == 'equaldist':
        output = composite_equaldist(rays, model, occgrid, transfer, chunksize, volume_origin, volume_phy, render_step_size)
    return output

def composite_equaldist(rays, model, occgrid, transfer, chunksize, volume_origin, volume_phy, render_step_size):
    is_dynamic = rays.shape[1] == 7
    keys = ['proj']
    if isinstance(model, PD_field):
        keys.extend(['prob_proj', 'dynamic_proj'])
    elif isinstance(model, SD_field):
        keys.extend(['static_proj', 'dynamic_proj'])
    elif isinstance(model, VPAL):
        keys.extend(['prob_proj', 'static_proj', 'dynamic_proj', 'gated_static_proj', 'gated_dynamic_proj'])

    split_rays = torch.split(rays, chunksize)
    render_result_chunks = {k: [] for k in keys}
    for ray_batch in split_rays:
        if_inter = if_intersect(ray_batch[:, :3], ray_batch[:, 3:], volume_origin, volume_phy)

        if if_inter:
            if is_dynamic:
                # train for nerf model if dynamic
                output_batch = batch_composite_equaldist_dynamic(ray_batch, model, occgrid, volume_origin, volume_phy, render_step_size)
            else:
                # render from given volume or train for nerf model if static
                output_batch = batch_composite_equaldist_static(ray_batch, model, transfer, volume_origin, volume_phy, render_step_size)
            
            for k in keys:
                render_result_chunks[k].append(output_batch[k])
        else:
            zero_tensor = torch.zeros_like(ray_batch[:, :1])
            for k in keys:
                render_result_chunks[k].append(zero_tensor)

    output = {k: torch.cat(render_result_chunks[k], dim=0) for k in keys}
    return output

def batch_composite_equaldist_dynamic(rays, model, occgrid, volume_origin, volume_phy, render_step_size):
    rays_origins = rays[:, :3]
    rays_dirs = rays[:, 3:6]
    timestamps = rays[:, 6:7]

    if occgrid is not None:
        ray_indices, t_start, t_end = occgrid.estimator.sampling(rays_o=rays_origins, rays_d=rays_dirs, render_step_size=render_step_size)
    else:
        ray_indices, t_start, t_end = sample_volume(rays_origins, rays_dirs, volume_origin, volume_phy, render_step_size)
    t_origins = rays_origins[ray_indices]
    t_dirs = rays_dirs[ray_indices]
    points = t_origins + t_dirs * (t_start + t_end)[:, None] / 2.0
    dists = (t_end - t_start)[..., None]   # [n_samples, 1]  weight for each sample value 

    points = ((points - volume_origin) / volume_phy)  # normalize between [0,1]  [n_samples, 3]
    time = timestamps[ray_indices]   # [n_samples, 1]
    points_time = torch.cat([points, time], dim=-1)   # [n_samples, 4]

    split_points_time = torch.split(points_time, 100000)
    
    point_keys = []
    proj_keys = []
    point_keys.extend(['weight_att', 'prob', 'static_att', 'dynamic_att', 'gated_static_att', 'gated_dynamic_att'])
    proj_keys.extend(['proj', 'prob_proj', 'static_proj', 'dynamic_proj', 'gated_static_proj', 'gated_dynamic_proj'])

    point_result_chunks = {k: [] for k in point_keys}
    point_result = {k: [] for k in point_keys}
    for pnts_t in split_points_time:
        ret = model(pnts_t)
        for k in point_keys:
            point_result_chunks[k].append(ret[k])

    point_result = {k: torch.cat(point_result_chunks[k], dim=0) for k in point_keys}

    n_rays = rays_origins.shape[0]
    output = {proj_keys[i]: volumetric_rendering_along_rays(dists, point_result[point_keys[i]], ray_indices, n_rays) for i in range(len(point_keys))}

    return output

def batch_composite_equaldist_static(rays, model, transfer, volume_origin, volume_phy, render_step_size):
    
    rays_origins = rays[:, :3]
    rays_dirs = rays[:, 3:6]
    
    ray_indices, t_start, t_end = sample_volume(rays_origins, rays_dirs, volume_origin, volume_phy, render_step_size)
    t_origins = rays_origins[ray_indices]
    t_dirs = rays_dirs[ray_indices]
    points = t_origins + t_dirs * (t_start + t_end)[:, None] / 2.0
    dists = (t_end - t_start)[..., None]   # [n_samples, 1]  weight for each sample value
     
    split_points = torch.split(points, 100000)
    val_all = []
    for pnts in split_points:
        if isinstance(model, nn.Module): # nerf model sampling
           pnts = ((pnts - volume_origin) / volume_phy)  # normalize between [0,1]
           ret = model(pnts)
           val_all.append(ret['weight_att'])

        else:  # volume sampling
           pnts = ((pnts - volume_origin) / volume_phy) * 2 - 1  # normalize between [-1,1]
           val_all.append(volume_sampling(pnts, model, transfer))

    att = torch.cat(val_all,dim=0)  # [n_samples, 1]  sample values

    n_rays = rays_origins.shape[0]
    proj = volumetric_rendering_along_rays(dists, att, ray_indices, n_rays)  # [n_rays, 1]

    output = {'proj': proj}

    return output

def volumetric_rendering_along_rays(weights, values, ray_indices, n_rays, method = 'index_add'):
    if weights is None:
        src = values
    else:
        src = weights * values  # [n_samples, 1]
    if method  == 'index_add':
        # index_add
        outputs = torch.zeros((n_rays, src.shape[-1]), device=src.device, dtype=src.dtype)  
        outputs.index_add_(0, ray_indices, src) # [n_rays, 1]
    elif method == 'segment_coo':
        # segment_coo
        outputs = segment_coo(
                src=src,
                index=ray_indices,
                out=torch.zeros([n_rays, src.shape[-1]], device=src.device, dtype=src.dtype),
                reduce='sum') # [n_rays, 1]
    return outputs

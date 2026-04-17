import torch
from util.util_func import img2nii
import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

def make_coords(volume_resolution, volume_phy, volume_origin, device):
    s1, s2, s3 = volume_phy
    o1, o2, o3 = volume_origin
    n1, n2, n3 = volume_resolution
    x = torch.linspace(o1, s1 + o1, n1 + 1, device=device, dtype=torch.float32)
    x_mid = (x[:-1] + x[1:]) / 2
    y = torch.linspace(o2, s2 + o2, n2 + 1, device=device, dtype=torch.float32)
    y_mid = (y[:-1] + y[1:]) / 2
    z = torch.linspace(o3, s3 + o3, n3 + 1, device=device, dtype=torch.float32)
    z_mid = (z[:-1] + z[1:]) / 2
    xyz = torch.meshgrid(x_mid, y_mid, z_mid)
    xyz = torch.stack(xyz, dim=-1)
    return xyz

def predict_volume_4d_VPAL(model, volume_resolution, volume_phy, volume_origin, time_sequence, 
                           show_indices, output_path, fusion_conf, out_other, device):
    
    dynamic_att_subpath = os.path.join(output_path, 'dynamic_att')
    os.makedirs(dynamic_att_subpath, exist_ok=True)
    gated_dynamic_att_subpath = os.path.join(output_path, 'gated_dynamic_att')
    os.makedirs(gated_dynamic_att_subpath, exist_ok=True)
    weight_att_subpath = os.path.join(output_path, 'weight_att')
    os.makedirs(weight_att_subpath, exist_ok=True)

    # temporal fusion mode
    if fusion_conf['mode'] == 'simple_avg':
        weighting_vector = np.ones_like(show_indices)
        weighting_vector = torch.tensor(weighting_vector, dtype=torch.float32, device=device)
        weighting_vector = weighting_vector / weighting_vector.sum()
    
    points = make_coords(volume_resolution.tolist(), volume_phy.tolist(), volume_origin.tolist(), device)
    volume_shape = points.shape[0:3]
    points = points.reshape(-1, 3)
    points = ((points - volume_origin) / volume_phy)  # normalize between [0, 1]
    split_points = torch.split(points, 100000)
    
    # reconstruct probability map and save it
    prob_all = []
    for pnts in split_points:
        prob_all.append(model.get_probability(pnts))
    prob_map = torch.cat(prob_all, dim=0)
    prob_map = prob_map.reshape(volume_shape)
    if out_other:
        img2nii(prob_map.transpose(0,2), os.path.join(output_path, 'probability.nii.gz'))

    # reconstruct static / gated static attenuation and save them
    static_att_all = []
    for pnts in split_points:
        static_att_all.append(model.get_static_att(pnts))
    static_att = torch.cat(static_att_all, dim=0)
    static_att = static_att.reshape(volume_shape)
    gated_static_att = (1-prob_map)*static_att
    if out_other:
        img2nii(static_att.transpose(0,2), os.path.join(output_path, 'static_att.nii.gz'))
        img2nii(gated_static_att.transpose(0,2), os.path.join(output_path, 'gated_static_att.nii.gz'))

    # reconstruct dynamic / gated dynamic / weight attenuation and save them
    dynamic_att_fused = 0
    gated_dynamic_att_fused = 0
    weight_att_fused = 0

    for i, index in enumerate(tqdm(show_indices, desc='Reconstructing')):
        weight = weighting_vector[i]
        time = time_sequence[index]
        dynamic_att_all = []
        gated_dynamic_att_all = []
        weight_att_all = []

        for pnts in split_points:
            pnts_t = torch.cat([pnts, time.expand([pnts.shape[0],1])], dim=1)
            ret = model(pnts_t)
            dynamic_att = ret['dynamic_att']
            gated_dynamic_att = ret['gated_dynamic_att']
            weight_att = ret['weight_att']
            dynamic_att_all.append(dynamic_att)
            gated_dynamic_att_all.append(gated_dynamic_att)
            weight_att_all.append(weight_att)
        
        dynamic_att_cur = torch.cat(dynamic_att_all, dim=0)
        dynamic_att_cur = dynamic_att_cur.reshape(volume_shape)
        
        gated_dynamic_att_cur = torch.cat(gated_dynamic_att_all, dim=0)
        gated_dynamic_att_cur = gated_dynamic_att_cur.reshape(volume_shape)

        weight_att_cur = torch.cat(weight_att_all, dim=0)
        weight_att_cur = weight_att_cur.reshape(volume_shape)
        
        dynamic_att_fused += dynamic_att_cur * weight
        gated_dynamic_att_fused += gated_dynamic_att_cur * weight
        weight_att_fused += weight_att_cur * weight
        if out_other:
            img2nii(dynamic_att_cur.transpose(0,2), os.path.join(dynamic_att_subpath, str(index).zfill(4)+'_dynamic_att.nii.gz'))
            img2nii(gated_dynamic_att_cur.transpose(0,2), os.path.join(gated_dynamic_att_subpath, str(index).zfill(4)+'_gated_dynamic_att.nii.gz'))
            img2nii(weight_att_cur.transpose(0,2), os.path.join(weight_att_subpath, str(index).zfill(4)+'_weight_att.nii.gz'))
    if out_other:
        img2nii(dynamic_att_fused.transpose(0,2), os.path.join(output_path, 'dynamic_att_'+fusion_conf['mode']+'.nii.gz'))
        img2nii(gated_dynamic_att_fused.transpose(0,2), os.path.join(output_path, 'gated_dynamic_att_'+fusion_conf['mode']+'.nii.gz'))
    img2nii(weight_att_fused.transpose(0,2), os.path.join(output_path, 'weight_att_'+fusion_conf['mode']+'.nii.gz'))
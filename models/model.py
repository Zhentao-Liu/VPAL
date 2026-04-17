import torch
import torch.nn as nn
import tinycudann as tcnn
import nerfacc

class occgrid(nn.Module):
    # remember modify nerfacc.estimator.occ_grid._update()
    def __init__(self, volume_origin, volume_phy, volume_resolution, s_rate=4):
        super().__init__()
        device = volume_origin.device
        roi_aabb = torch.cat([volume_origin, volume_origin + volume_phy], dim=0)
        down_sampled_resolution = (volume_resolution / s_rate).int()
        self.estimator = nerfacc.OccGridEstimator(roi_aabb=roi_aabb, 
                                                  resolution=down_sampled_resolution).to(device)
                                                
class VPAL(nn.Module):
    def __init__(self, conf, ):
        super().__init__()
        self.conf = conf
        self.probgrid = tcnn.Encoding(n_input_dims=3, encoding_config=conf.probgrid, dtype=torch.float32)
        self.hash3dgrid = tcnn.Encoding(n_input_dims=3, encoding_config=conf.hash3dgrid, dtype=torch.float32)
        self.hash4dgrid = tcnn.Encoding(n_input_dims=4, encoding_config=conf.hash4dgrid, dtype=torch.float32)
        self.probnet = tcnn.Network(n_input_dims=self.probgrid.n_output_dims, n_output_dims=1, network_config=conf['probnet'])
        self.net3d = tcnn.Network(n_input_dims=self.hash3dgrid.n_output_dims, n_output_dims=1, network_config=conf['net3d'])
        self.net4d = tcnn.Network(n_input_dims=self.hash4dgrid.n_output_dims, n_output_dims=1, network_config=conf['net4d'])

        if self.conf.coarse2fine.enabled:
            self.step_prob = self.conf.coarse2fine.step_prob
            self.step_3d = self.conf.coarse2fine.step_3d
            self.step_4d = self.conf.coarse2fine.step_4d

    def set_active_level_prob(self, current_iter):
        add_levels_prob = current_iter // self.step_prob
        add_levels_prob = min(self.conf.probgrid.n_levels - self.conf.coarse2fine.init_active_level_prob, add_levels_prob)
        self.active_levels_prob = self.conf.coarse2fine.init_active_level_prob + add_levels_prob
    
    def set_active_level_3d(self, current_iter):
        add_levels_3d = current_iter  // self.step_3d
        add_levels_3d = min(self.conf.hash3dgrid.n_levels - self.conf.coarse2fine.init_active_level_3d, add_levels_3d)
        self.active_levels_3d = self.conf.coarse2fine.init_active_level_3d + add_levels_3d
    
    def set_active_level_4d(self, current_iter):
        add_levels_4d = current_iter // self.step_4d
        add_levels_4d = min(self.conf.hash4dgrid.n_levels - self.conf.coarse2fine.init_active_level_4d, add_levels_4d)
        self.active_levels_4d = self.conf.coarse2fine.init_active_level_4d + add_levels_4d

    @torch.no_grad()
    def _get_coarse2fine_mask(self, encoding, active_levels, feat_dim):
        mask = torch.zeros_like(encoding)
        mask[..., :(active_levels)*feat_dim] = 1
        return mask
    
    def get_probability(self, xyz):
        prob_feat = self.probgrid(xyz)
        if self.conf.coarse2fine.enabled:
            prob_mask = self._get_coarse2fine_mask(prob_feat, self.active_levels_prob, self.conf.probgrid.n_features_per_level)
            prob_feat = prob_feat * prob_mask
        probability = self.probnet(prob_feat).to(torch.float32)
        return probability
    
    def get_static_att(self, xyz):
        static_feat = self.hash3dgrid(xyz)
        if self.conf.coarse2fine.enabled:
            static_mask = self._get_coarse2fine_mask(static_feat, self.active_levels_3d, self.conf.hash3dgrid.n_features_per_level)
            static_feat = static_feat * static_mask
        static_att = self.net3d(static_feat).to(torch.float32)
        return static_att
    
    def get_dynamic_att(self, xyzt):
        dynamic_feat = self.hash4dgrid(xyzt)
        if self.conf.coarse2fine.enabled:
            dynamic_mask = self._get_coarse2fine_mask(dynamic_feat, self.active_levels_4d, self.conf.hash4dgrid.n_features_per_level)
            dynamic_feat = dynamic_feat * dynamic_mask
        dynamic_att = self.net4d(dynamic_feat).to(torch.float32)
        return dynamic_att

    def forward(self, xyzt, ):
        xyz = xyzt[:, :3]
        probability = self.get_probability(xyz)
        static_att = self.get_static_att(xyz)
        dynamic_att = self.get_dynamic_att(xyzt)
        gated_static_att = (1-probability) * static_att
        gated_dynamic_att = probability * dynamic_att
        weight_att = gated_static_att + gated_dynamic_att
        ret={'prob':probability, 'static_att':static_att, 'dynamic_att':dynamic_att, 
             'gated_static_att':gated_static_att, 'gated_dynamic_att':gated_dynamic_att, 'weight_att':weight_att}
        return ret
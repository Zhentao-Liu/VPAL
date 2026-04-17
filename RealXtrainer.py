import os
import torch
from models.render import composite
from models.reconstruct import *
from models.loss import loss_fn, prob_reg_loss, prob_entropy_loss
from models.model import occgrid
from util.util_func import *
import datetime
from tqdm import tqdm
import time as timeloger
from tensorboardX import SummaryWriter

class RealXtrainer():
    def __init__(self, model, data, args, conf, device):
        self.model = model
        model_params = count_paras_M(model)
        model_params_log = f"Total number of model parameters: {model_params :.2f} M"
        print(model_params_log)
        self.data = data
        self.args = args
        self.conf = conf
        self.device = device

        # volume_info
        self.volume_phy = torch.tensor(data.camera_paras['volume_phy'], dtype=torch.float32, device=device)
        self.volume_origin = -0.5 * self.volume_phy
        self.volume_spacing = torch.min(torch.tensor(data.camera_paras['volume_spacing'], dtype=torch.float32, device=device))
        self.volume_resolution = torch.tensor(data.camera_paras['volume_resolution'], dtype=torch.int64, device=device)
        self.use_occgrid = conf['occgrid'].enabled
        if self.use_occgrid:
            self.occgrid = occgrid(self.volume_origin, self.volume_phy, self.volume_resolution, conf.occgrid.s_rate)
            self.occ_thres = conf.occgrid.thres
            self.occ_update_n = conf.occgrid.updata_n
            occ_params = count_paras(self.occgrid)
            occ_params_log = f"Total number of occgrid parameters: {occ_params:.4f} MB"
            print(occ_params_log)
        else:
            self.occgrid = None

        # render
        self.ray_seen = conf.get_int('render.ray_seen')
        self.factor = conf.get_float('render.factor')
        self.render_step_size = self.factor * self.volume_spacing 
        self.sample_mode = conf.get('render.sample_mode')
        self.chunksize = conf.get_int('render.chunksize')

        # init render stepup
        self.render_kwargs = dict(
            sample_mode = self.sample_mode,
            chunksize = self.chunksize,
            volume_origin = self.volume_origin,
            volume_phy = self.volume_phy,
            render_step_size = self.render_step_size,
        )

        # fusion
        self.fusion_conf = conf['fusion']

        # flow-consistency
        self.flow_con_enabled = conf.get('flow_consistency.enabled')
        self.perturb_rays = conf.get_int('flow_consistency.perturb_rays')
        self.TP_std = conf.get('flow_consistency.TP_std')
        self.beta = conf.get('flow_consistency.beta')

        # interval
        self.print_interval = conf.get_int('print.print_interval')
        self.save_interval = conf.get_int('print.save_interval')
        self.vis_interval = conf.get_int('print.vis_interval')

        # epoch
        self.num_epochs = args.epochs
        self.history_epoch = args.history_epoch

        # others
        self.is_train = args.is_train
        self.disable_2d = args.disable_2d
        self.disable_fixview = args.disable_fixview
        self.fixview = args.fixview
        self.disable_fixtime = args.disable_fixtime
        self.fixtime = args.fixtime
        self.disable_3d = args.disable_3d

        # logs
        self.logs_path = os.path.join(args.output_path, args.name, 'logs')
        os.makedirs(self.logs_path, exist_ok=True)
        f_exp = open(self.logs_path + '/exp_state.txt', mode='a')
        f_exp.write(model_params_log + '\n')
        if self.use_occgrid:
            f_exp.write(occ_params_log + '\n')
        f_exp.close()

        self.visuals_path = os.path.join(args.output_path, args.name, 'visuals')
        os.makedirs(self.visuals_path, exist_ok=True)
        self.checkpoints_path = os.path.join(args.output_path, args.name, 'checkpoints')
        os.makedirs(self.checkpoints_path, exist_ok=True)

        # loss
        self.prob_reg_weight = conf.get_float('loss.prob_reg_weight')

        # optimizer
        self.optim_nerf = get_optimizer(conf.optim_nerf, self.model)
        self.lrsched_nerf = get_scheduler(conf.optim_nerf, self.optim_nerf)

        # load weights & optimizer & iterator
        self.begin_epochs = 0
        os.makedirs("%s/ckpt_history" % (self.checkpoints_path,), exist_ok=True)
        self.latest_model_path = "%s/ckpt_latest" % (self.checkpoints_path,)
        self.history_model_path = "%s/ckpt_history/ckpt_" % (self.checkpoints_path,)
        if args.resume:
            self.load_ckpt(self.history_epoch)
        
    def save_ckpt(self, epoch):
        sd_file = {
            'iter': epoch + 1,
            'model': self.model.state_dict(),
            'optim_nerf': self.optim_nerf.state_dict(),
            'lrsched_nerf': self.lrsched_nerf.state_dict(),
        }
        if self.use_occgrid:
            sd_file['occgrid'] = self.occgrid.state_dict()
        torch.save(sd_file, self.latest_model_path)
        torch.save(sd_file, self.history_model_path + str(epoch))

    def load_ckpt(self, history_epoch):
        sd_file = None
        if history_epoch == 0:
            if os.path.exists(self.latest_model_path):
                sd_file = torch.load(self.latest_model_path, map_location=self.device)
        else:
            if os.path.exists(self.history_model_path+str(history_epoch)):
                sd_file = torch.load(self.history_model_path+str(history_epoch), map_location=self.device)
        if sd_file is not None:
            if 'iter' in sd_file: self.begin_epochs = sd_file['iter']
            if 'model' in sd_file: self.model.load_state_dict(sd_file['model'])
            if self.use_occgrid and 'occgrid' in sd_file:
                self.occgrid.load_state_dict(sd_file['occgrid'])
            if self.is_train:
                if 'optim_nerf' in sd_file: self.optim_nerf.load_state_dict(sd_file['optim_nerf'])
                if 'lrsched_nerf' in sd_file: self.lrsched_nerf.load_state_dict(sd_file['lrsched_nerf'])

    def train_step(self, data, epoch):
        
        loss_dict = {}

        device = self.device

        train_indice = data.train_indice
        
        index = train_indice
        rays = data.rays_train[index]
        inds_range = self.args.nviews * data.H * data.W
        
        rays = rays.reshape(-1, rays.shape[-1])
        proj = data.proj_train[index] # 30 projections [30, 1240, 960]
        proj = proj.reshape(-1, 1)
        pix_inds = torch.randint(0, inds_range, (self.ray_seen,))
        pix_gt = proj[pix_inds]
        pix_rays = rays[pix_inds] # [ray_seen, 6]  r = o + t*d

        if self.use_occgrid:
            self.occgrid.estimator.update_every_n_steps(
                step=epoch,
                occ_eval_fn=self.model.get_probability,
                occ_thre=self.occ_thres,
                n=self.occ_update_n
            )

        loss_weight = 1

        proj_time = data.proj_time[index]  
        proj_time = proj_time.reshape(-1, 1)
        pix_time = proj_time[pix_inds]

        if self.flow_con_enabled:
            pix_time_perturb = torch.randn(self.ray_seen, 1).to(device) * ( self.TP_std/self.args.nviews)
            pix_time_perturb[self.perturb_rays:] = 0   # only partial rays receive time perturbations
            loss_dict["temporal_perturbation_avg"] = round(torch.mean(torch.abs(pix_time_perturb[:self.perturb_rays])).item(), 8)
            pix_time = pix_time + pix_time_perturb
            pix_time = torch.clamp(pix_time, 0, 1)
            pix_render_input = torch.cat((pix_rays, pix_time),dim=-1)  # [ray_seen, 7]
            loss_weight = loss_weight / (1 + self.beta * (self.args.nviews/(3*self.TP_std))  * torch.abs(pix_time_perturb))

        else:
            pix_render_input = torch.cat((pix_rays, pix_time),dim=-1)  # [ray_seen, 7]

        output = composite(rays=pix_render_input, model=self.model, occgrid=self.occgrid, transfer=False, **self.render_kwargs)
        
        loss_pix = loss_fn(output, pix_gt, loss_weight)
        
        loss = loss_pix

        use_prob_reg = self.prob_reg_weight > 0

        if use_prob_reg:
            prob_reg = prob_reg_loss(self.model, 10000, device)
            loss = loss + prob_reg * self.prob_reg_weight              

        loss.backward()
        self.optim_nerf.step()
        self.optim_nerf.zero_grad()

        mem_reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024**3)
        loss_dict['mem_mb'] = round(mem_reserved_gb, 2)

        loss_dict['loss_pix'] = round(loss_pix.item(), 8)

        if use_prob_reg:
            loss_dict['prob_reg'] = round(prob_reg.item(), 8)

        loss_dict['loss'] = round(loss.item(), 8)

        loss_dict["proj_gt_min"] = round(torch.min(pix_gt).item(), 4)
        loss_dict["proj_gt_max"] = round(torch.max(pix_gt).item(), 4)

        loss_dict["proj_pred_min"] = round(torch.min(output['proj']).item(), 4)
        loss_dict["proj_pred_max"] = round(torch.max(output['proj']).item(), 4)

        if self.model.conf.coarse2fine.enabled: 
            
            loss_dict["active_levels_prob"] = self.model.active_levels_prob
            loss_dict["active_levels_3d"] = self.model.active_levels_3d
            loss_dict["active_levels_4d"] = self.model.active_levels_4d

        return loss_dict

    def vis_step(self, data, epoch):

        device = self.device

        if self.use_occgrid:
            print('*** Saving occupancy ***')
            save_occgrid(self.occgrid.estimator.binaries, os.path.join(self.visuals_path, 'binarygrid.nii.gz'))

        loss_dict = {}

        H, W = data.H, data.W
        
        # 2D projection visualization
        print('*** 2D projections visualization ***')

        # contrast adjust

        keys = ['proj']
        if self.out_other:
            keys.extend(['prob_proj', 'static_proj', 'dynamic_proj', 'gated_static_proj', 'gated_dynamic_proj'])

        if not self.disable_2d:
            psnrs = []
            ssims = []
            results_list = {k: [] for k in keys}

            for index in tqdm(range(data.Nviews), desc='Rendering rotation view'):

                rays = data.rays_clean[index:index + 1]
                proj_clean = data.proj_clean[index, :, :]
                rays = rays.reshape(-1, rays.shape[-1])

                time = data.proj_time[index]
                time = time.reshape(-1, 1)
                render_input = torch.concat([rays, time], dim=-1)   # cam_origin, cam_dirs, time

                output = composite(rays=render_input, model=self.model, occgrid=self.occgrid, transfer=False, **self.render_kwargs)               
                
                for k in keys:
                    results_list[k].append(output[k].reshape(H, W))
                
                proj_predict = output['proj'].reshape(H, W)
                
                psnr_cur = get_psnr(data_normal(proj_clean), data_normal(proj_predict))
                ssim_cur = get_ssim_2d(data_normal(proj_clean), data_normal(proj_predict))
                psnrs.append(psnr_cur)
                ssims.append(ssim_cur)
            
            loss_dict['psnr_eval'] = round(torch.tensor(psnrs)[data.eval_indice].mean().item(), 8) if len(data.eval_indice)!=0 else 0
            loss_dict['psnr_train'] = round(torch.tensor(psnrs)[data.train_indice].mean().item(), 8) if len(data.train_indice)!=0 else 0
            loss_dict['psnr_all'] = round(torch.tensor(psnrs).mean().item(), 8)
            loss_dict['ssim_eval'] = round(torch.tensor(ssims)[data.eval_indice].mean().item(), 8) if len(data.eval_indice)!=0 else 0
            loss_dict['ssim_train'] = round(torch.tensor(ssims)[data.train_indice].mean().item(), 8) if len(data.train_indice)!=0 else 0
            loss_dict['ssim_all'] = round(torch.tensor(ssims).mean().item(), 8)
            
            proj_predict_path = os.path.join(self.visuals_path, 'proj', str(epoch), 'rotationview')
            os.makedirs(proj_predict_path, exist_ok=True)
            for k in keys:
                results_list[k] = torch.stack(results_list[k], dim=0)
                img2nii(results_list[k], os.path.join(proj_predict_path, k+'.nii.gz'))
        

        if not self.disable_fixview:
            results_list = {k: [] for k in keys}
            for index in tqdm(range(data.Nviews), desc='Rendering fix view'):
                rays = data.rays_clean[self.fixview:self.fixview + 1]
                rays = rays.reshape(-1, rays.shape[-1])
                time = data.proj_time[index]
                time = time.reshape(-1, 1)
                render_input = torch.concat([rays, time], dim=-1)   # cam_origin, cam_dirs, time

                output = composite(rays=render_input, model=self.model, occgrid=self.occgrid, transfer=False, **self.render_kwargs)   
                
                for k in keys:
                    results_list[k].append(output[k].reshape(H, W))

            proj_predict_path = os.path.join(self.visuals_path, 'proj', str(epoch), f'fixview_{self.fixview}')
            os.makedirs(proj_predict_path, exist_ok=True)
            for k in keys:
                results_list[k] = torch.stack(results_list[k], dim=0)
                img2nii(results_list[k], os.path.join(proj_predict_path, k+'.nii.gz'))
        
        if not self.disable_fixtime:
            results_list = {k: [] for k in keys}
            for index in tqdm(range(data.Nviews), desc='Rendering fix time'):
                rays = data.rays_clean[index:index+1]
                rays = rays.reshape(-1, rays.shape[-1])
                time = torch.full((rays.shape[0], 1), self.fixtime/data.Nviews, dtype=torch.float32, device='cuda')
                render_input = torch.concat([rays, time], dim=-1)
                
                output = composite(rays=render_input, model=self.model, occgrid=self.occgrid, transfer=False, **self.render_kwargs)

                for k in keys:
                    results_list[k].append(output[k].reshape(H, W))
                
            proj_predict_path = os.path.join(self.visuals_path, 'proj', str(epoch), f'fixtime_{self.fixtime}')
            os.makedirs(proj_predict_path, exist_ok=True)
            for k in keys:
                results_list[k] = torch.stack(results_list[k], dim=0)
                img2nii(results_list[k], os.path.join(proj_predict_path, k+'.nii.gz'))
                
        if not self.disable_3d:
            # volume visualization
            print('*** 3D reconstruction visualization ***')
            volume_predict_path = os.path.join(self.visuals_path, 'volume', str(epoch))
            os.makedirs(volume_predict_path, exist_ok=True)
            time_sequence = data.proj_time[:, 0, 0]

            predict_volume_4d_VPAL(self.model, self.volume_resolution, self.volume_phy, self.volume_origin, time_sequence, data.all_indice, 
                                    volume_predict_path, self.fusion_conf, self.args.out_other, device)

        # 测试时候的内存占用
        mem_reserved_gb = torch.cuda.max_memory_reserved(self.device) / (1024**3)
        loss_dict['mem_mb'] = round(mem_reserved_gb, 2)

        return loss_dict

    def epoch_init(self, model, epoch):
        # set active hash table level
        if model.conf.coarse2fine.enabled:
            model.set_active_level_prob(epoch)
            model.set_active_level_3d(epoch)
            model.set_active_level_4d(epoch)

    def start(self):

        # train stage
        now = datetime.datetime.now()
        begin_exp = 'Experiment Start: ' + now.strftime('%Y-%m-%d %H:%M:%S') + '\n'
        f_exp = open(self.logs_path + '/exp_state.txt', mode='a')
        f_exp.write(begin_exp)
        f_exp.close()

        if self.is_train:

            writer = SummaryWriter(log_dir=self.logs_path)

            exp_start_time = timeloger.time()

            for epoch in tqdm(range(self.begin_epochs, self.num_epochs), desc='Training'):

                self.epoch_init(self.model, epoch)
  
                # network training
                self.model.train()
                if self.use_occgrid:
                    self.occgrid.estimator.train()
                train_loss = self.train_step(self.data, epoch)
                tblog(writer, train_loss, epoch)

                train_loss_str = fmt_loss_str(train_loss)

                if epoch % self.print_interval == 0 or epoch == self.num_epochs - 1:
                    now = datetime.datetime.now()
                    f_train_ls = open(self.logs_path + '/train_ls.txt', mode='a')
                    # f_train_ls.write(
                    #     now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + train_loss_str + ' lr:' + 
                    #     str(self.optim_nerf.param_groups[0]['lr']) + '\n')
                    f_train_ls.write(
                        now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + train_loss_str + '\n'
                    )
                    f_train_ls.close()

                # network saving
                if ((epoch % self.save_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    print('***** Network & Optimizer Saving *****')
                    self.save_ckpt(epoch)

                # lr schedule
                self.lrsched_nerf.step()

                if epoch == self.num_epochs - 1:
                    train_end_time = timeloger.time()
                    train_elapsed_time = train_end_time - exp_start_time
                    train_h, train_m, train_s = int(train_elapsed_time // 3600), int((train_elapsed_time % 3600) //60), int(train_elapsed_time % 60)
                    elapsed_time_str = f"{train_h:02}:{train_m:02}:{train_s:02}"
                    f_exp = open(self.logs_path + '/exp_state.txt', mode='a')
                    f_exp.write(f"Training elapsed time: {elapsed_time_str}\n")
                    f_exp.close()

                # results visualization
                if epoch % self.vis_interval == self.vis_interval-1 or epoch == self.num_epochs - 1:
                    print('***** Reconstruction Results Visualization *****')
                    os.makedirs(self.visuals_path + '/proj', exist_ok=True)
                    os.makedirs(self.visuals_path + '/volume', exist_ok=True)
                    self.model.eval()
                    if self.use_occgrid:
                        self.occgrid.estimator.train()
                    with torch.no_grad():
                        vis_loss = self.vis_step(self.data, epoch)
                    vis_loss_str = fmt_loss_str(vis_loss)
                    now = datetime.datetime.now()
                    f_vis_ls = open(self.logs_path + '/vis_ls.txt', mode='a')
                    f_vis_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + vis_loss_str + '\n')
                    f_vis_ls.close()
                    print('*** visualization:', now.strftime('%Y-%m-%d %H:%M:%S'), 'Epoch:', str(epoch), vis_loss_str,)
            
            # writer.close()
        
        # visualization when not training (finish training)
        else:
            print('***** Reconstruction Results Visualization *****')
            
            epoch = self.begin_epochs

            self.epoch_init(self.model, epoch)

            os.makedirs(self.visuals_path + '/proj', exist_ok=True)
            os.makedirs(self.visuals_path + '/volume', exist_ok=True)
            self.model.eval()
            if self.use_occgrid:
                self.occgrid.estimator.eval()
            with torch.no_grad():
                vis_loss = self.vis_step(self.data, epoch)
            vis_loss_str = fmt_loss_str(vis_loss)
            now = datetime.datetime.now()
            f_vis_ls = open(self.logs_path + '/vis_ls.txt', mode='a')
            f_vis_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' visualization:' + vis_loss_str + '\n')
            f_vis_ls.close()
            print('*** visualization:', now.strftime('%Y-%m-%d %H:%M:%S'), vis_loss_str,)
        
        now = datetime.datetime.now()
        end_exp = 'Experiment End: ' + now.strftime('%Y-%m-%d %H:%M:%S') + '\n'
        f_exp = open(self.logs_path + '/exp_state.txt', mode='a')
        f_exp.write(end_exp)
        f_exp.close()
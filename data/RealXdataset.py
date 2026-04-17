import numpy as np
import torch.utils.data
import os
import json
from models.render import get_rays, angle2vec, get_rays_vec
import SimpleITK as sitk

def add_realistic_noise(projections_, I0=1e5, gaussian_std=10):
    """
    为 DSA/CT 投影添加符合物理规律的噪声 (Poisson + Gaussian)
    
    Args:
        projections: 原始投影数据 (Attenuation values)
        I0: 入射光子数 (模拟剂量水平，值越大噪声越小，通常 1e4 - 1e6)
        gaussian_std: 电子噪声的标准差 (通常 5-20)
    """
    
    projections = np.clip(projections_.copy(), 0, None)

    # 1. 【域转换】 Attenuation -> Photon Counts (Intensity)
    # TIGRE 源码里有个 normalization (projections / max_proj)，
    # 如果你的投影已经是标准的 -ln(I/I0)，其实不需要除以 max_proj。
    # 但为了保持和 TIGRE 逻辑一致，或者防止数据没归一化，我们保留这个缩放逻辑：
    max_proj = np.max(projections)
    if max_proj == 0:
        return projections # 避免除0
        
    # 计算理论上的光子数 (Expected counts)
    expected_counts = I0 * np.exp(-projections / max_proj)

    # 2. 【核心步骤】 添加物理噪声
    # 2.1 泊松噪声 (Photon Shot Noise): 信号本身服从泊松分布
    # np.random.poisson 返回的是整数，转为 float 进行后续计算
    noisy_counts = np.random.poisson(expected_counts).astype(np.float32)

    # 2.2 高斯噪声 (Electronic Noise): 加性噪声
    gaussian_noise = np.random.normal(0, gaussian_std, size=projections.shape)
    noisy_counts += gaussian_noise

    # 3. 【数值保护】 防止 log(<=0)
    # 因为高斯噪声可能导致光子数变为负数，或者泊松采样为0
    # 真实的探测器会有暗电流或读出截断，这里我们设一个极小值防止 NaN
    noisy_counts = np.maximum(noisy_counts, 0.1)

    # 4. 【域转换】 Photon Counts -> Attenuation
    # 还原回投影域
    noisy_projections = -np.log(noisy_counts / I0) * max_proj

    return noisy_projections.astype(np.float32)


class RealXdataset(torch.utils.data.Dataset):
    """
    Dataset for sparse view DSA reconstruction (real projection of vessel dataset)
    """
    def __init__(self, args, device):
        super().__init__()
        
        data_dir = args.datadir
        train_views = args.nviews
        geotype = args.geotype


        proj_path = os.path.join(data_dir, 'proj.nii.gz')
        print('Loading DSA series, file_name:', proj_path)

        proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))

        proj_clean = proj.copy()
        proj_train = proj.copy()

        if args.noise_aug:
            proj_train = add_realistic_noise(proj_train, args.I0, args.GSstd)
        
        proj_clean = np.clip(proj_clean, 0, None)
        proj_train = np.clip(proj_train, 0, None)

        self.proj_clean = torch.tensor(proj_clean, dtype=torch.float32, device=device)
        self.proj_train = torch.tensor(proj_train, dtype=torch.float32, device=device)

        json_path = os.path.join(data_dir, 'transforms.json')
        print('Loading corresponding transforms.json:', json_path)
        with open(json_path) as f:
            self.camera_paras = json.load(f)

        self.volume_res = self.camera_paras['volume_resolution']
        self.Nviews = self.camera_paras['N_views']
        self.sad = self.camera_paras['sad']

        self.sid = self.camera_paras['sid']
        self.W, self.H = self.camera_paras['proj_resolution']
        self.proj_spacing = torch.tensor(self.camera_paras['proj_spacing'], dtype=torch.float32, device=device)

        if geotype == 'matrix':
            poses = []
            for i in range(self.Nviews):
                frame = self.camera_paras['frames'][i]
                pose = torch.tensor(frame['extrinsics'], dtype=torch.float32, device=device)
                poses.append(pose)
            self.poses = torch.stack(poses)
            self.fx = self.sid/self.proj_spacing[1]
            self.fy = self.sid/self.proj_spacing[0]
            self.rays = get_rays(self.poses, self.H, self.W, self.fx, self.fy).to(device)
        elif geotype == 'vec':
            vecs_clean = []
            PrimaryAngles_clean = []
            for i in range(self.Nviews):
                frame = self.camera_paras['frames'][i]
                PrimaryAngle = frame['PrimaryAngle']
                PrimaryAngles_clean.append(PrimaryAngle)
                vec = angle2vec(PrimaryAngle, 0, [0,0,0], self.sid, self.sad, self.proj_spacing[0].item(), self.proj_spacing[1].item())
                vecs_clean.append(vec)
            vecs_clean = np.stack(vecs_clean)
            PrimaryAngles_clean = np.stack(PrimaryAngles_clean)
            self.PrimaryAngles_clean = PrimaryAngles_clean
            self.vecs_clean = torch.tensor(vecs_clean, dtype=torch.float32, device=device)
            self.rays_clean = get_rays_vec(self.vecs_clean, self.H, self.W).to(device)

            vecs_train = []
            PrimaryAngles_train = []
            for i in range(self.Nviews):
                PrimaryAngle = self.PrimaryAngles_clean[i]
                
                if args.angle_aug:
                    delta_deg = np.random.uniform(
                        -args.angle_aug_size,
                        args.angle_aug_size
                    )
                    delta_rad = np.deg2rad(delta_deg)
                    PrimaryAngle = PrimaryAngle + delta_rad
                
                PrimaryAngles_train.append(PrimaryAngle)
                vec = angle2vec(PrimaryAngle, 0, [0,0,0], self.sid, self.sad, self.proj_spacing[0].item(), self.proj_spacing[1].item())
                vecs_train.append(vec)
            vecs_train = np.stack(vecs_train)
            PrimaryAngles_train = np.stack(PrimaryAngles_train)
            self.PrimaryAngles_train = PrimaryAngles_train
            self.vecs_train = torch.tensor(vecs_train, dtype=torch.float32, device=device)
            self.rays_train = get_rays_vec(self.vecs_train, self.H, self.W).to(device)


        self.proj_time = torch.arange(0, self.Nviews, 1, device=device) / self.Nviews
        self.proj_time = self.proj_time.unsqueeze(1).unsqueeze(2).expand_as(self.proj_clean)

        all = np.arange(self.Nviews)
        self.all_indice = all
        self.train_indice = np.arange(0, self.Nviews, self.Nviews/train_views).astype(int)   # note that train views could not be zero
        if self.train_indice[-1] >= self.Nviews:
            self.train_indice[-1] = self.Nviews - 1
        # self.train_indice = np.linspace(0, self.Nviews-1, train_views).astype(int)
        self.eval_indice = np.delete(all, self.train_indice).astype(int)

        print('train views:', self.train_indice)
        print('eval views:', self.eval_indice)


        if args.noise_aug:
            print(f"[Aug] Noise enabled: I0 = {args.I0}, GSstd = {args.GSstd}")
        
        if args.angle_aug:
            print(f"[Aug] Angle perturbation enabled: ±{args.angle_aug_size} deg")


    def __len__(self):
        return self.Nviews   # number of all projections


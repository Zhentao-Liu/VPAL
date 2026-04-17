import math
import torch
import SimpleITK as sitk
import numpy as np
from skimage.metrics import structural_similarity
import pydicom
import imageio

def fmt_loss_str(losses):
    return (" " + " ".join(k + ":" + str(losses[k]) for k in losses))

def data_normal(data):
    min = data.min()
    max = data.max()
    data = (data - min) / (max - min)
    return data

def img2nii(img, dst_nii, spacing=None, origin=None):
    if torch.is_tensor(img):
        img = img.cpu().detach().numpy()
    image = sitk.GetImageFromArray(img)
    if spacing is not None:
        image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    sitk.WriteImage(image, dst_nii)

class Averager:
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    def add(self, value):
        self.sum += value
        self.count += 1
    
    def avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count

def get_psnr(pred, target):
    """
    Compute PSNR of two tensors (2D/3D) in decibels.
    pred/target should be of same size or broadcastable
    The max intensity should be 1, thus, it's better
    to normalize into [0,1]
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    mse = ((pred - target) ** 2).mean()
    if mse!=0:
      psnr = -10 * math.log10(mse)
    else:
      psnr = 'INF'
    return psnr

def get_ssim_2d(pred, target, data_range=1.0):
    """
    Compute SSIM of two tensors (2D) in decibels.
    pred/target should be of same size or broadcastable
    The max intensity should be 1, thus, it's better
    to normalize into [0,1]
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    ssim = structural_similarity(pred, target, data_range=data_range)
    return ssim

def get_ssim_3d(arr1, arr2, size_average=True, data_range=None):
    '''
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    '''
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = structural_similarity(arr1_d[i], arr2_d[i], data_range=data_range)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = structural_similarity(arr1_h[i], arr2_h[i], data_range=data_range)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    ssim_w = []
    for i in range(N):
        ssim = structural_similarity(arr1[i], arr2[i], data_range=data_range)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg
    
def linear_conversion(x, c, w, ymin=0.0, ymax=255.0):
    '''
    Apply linear conversion when VOI LUT Function is absent or has a value of LINEAR
    Input:
      x: DICOM pixel array after applying rescale slope and intercept, np.float32
      c: Window center, float
      w: Window width, float
      ymin: Minimum intensity for display, float
      ymax: Maximum intensity for display, float
    Output:
      y: Array with the same shape as x after linear conversion, np.float32
    '''
    y = np.full_like(x, ymin,)
    low = c - 0.5 - (w - 1) / 2
    high = c - 0.5 + (w - 1) / 2
    y[x <= low] = ymin
    y[x > high] = ymax
    mask = (x > low) & (x < high) 
    y[mask] = ((x[mask] - (c - 0.5)) / (w-1) + 0.5)* (ymax - ymin) + ymin
    return y

def dicomread(dicom_path, converted=False):
    dicomfile = pydicom.dcmread(dicom_path)
    # 获取Rescale Slope和Rescale Intercept
    rescale_slope = dicomfile.RescaleSlope if hasattr(dicomfile, 'RescaleSlope') else 1.0
    rescale_intercept = dicomfile.RescaleIntercept if hasattr(dicomfile, 'RescaleIntercept') else 0.0
    # 将像素数组转换为浮点数类型，并应用Rescale Slope和Rescale Intercept，从而恢复pixel_array读取由于取整导致的负数截断误差
    dataarray = dicomfile.pixel_array.astype(np.float32) * rescale_slope + rescale_intercept
    if dicomfile.VOILUTFunction == 'LINEAR' and converted:
        dataarray = linear_conversion(dataarray, float(dicomfile.WindowCenter), float(dicomfile.WindowWidth))
    return dataarray

def get_optimizer(conf, model):

    if conf.type == 'Adam':
        weight_decay = conf.params.weight_decay

        lr_p = conf.params.lr_p
        lr_s = conf.params.lr_s
        lr_d = conf.params.lr_d
        param_groups = [
            {
                'params': list(model.probgrid.parameters()) + list(model.probnet.parameters()),
                'lr': lr_p
            },
            {
                'params': list(model.hash3dgrid.parameters()) + list(model.net3d.parameters()),
                'lr': lr_s
            },
            {
                'params': list(model.hash4dgrid.parameters()) + list(model.net4d.parameters()),
                'lr': lr_d
            }
        ]        

        optim = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    return optim

def get_scheduler(conf, optim):
    
    if conf.sched.type == 'step':
        
        step_size = conf.sched.step_param.step_size
        gamma = conf.sched.step_param.gamma
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    
    return scheduler

def set_random_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def count_paras(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)  # Convert to MB

def count_paras_M(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def tblog(writer, loss, epoch):
    for key, value in loss.items():
        writer.add_scalar(key, value, global_step=epoch)

def curve_contrast_adjustment(image, curve_points):
    adjusted_image = np.interp(image, curve_points[:, 0], curve_points[:, 1])
    return adjusted_image

def array2video(images, output_path, fps):
    '''
    Convert an array of images to a video file.
    :param images: Numpy array of shape [frames, height, width, channels], float in range [0, 1]
    :param output_path: Path where the video file will be saved.
    :param fps: Frames per second of the output video.
    '''
    if torch.is_tensor(images):
        images = images.cpu().detach().numpy()
    images = (images*255).astype(np.uint8)
    writer = imageio.get_writer(output_path, fps=fps)  # 注意fps是作为关键字参数传递
    for image in images:
        writer.append_data(image)
    writer.close()

def img2video(img, path, fps):
    curve_points = np.array([[0.0, 0.0],
                             [0.5, 0.5],
                             [1.0, 1.0]])
    adjusted_array = curve_contrast_adjustment(img, curve_points)
    array2video(adjusted_array, path, fps)

def save_occgrid(binarygrid, output_path):
    binarygrid = binarygrid.squeeze().cpu().detach().numpy().astype(np.int8).transpose(2, 0, 1) + 1e-6
    binarygrid = np.rot90(binarygrid, k=1, axes=(1, 2))
    binarygrid = np.flip(binarygrid, axis=1)
    img2nii(binarygrid, output_path)    
import torch
import torch.nn.functional as F

def loss_cal(pred, gt, weight, type):
    if type == 'L1':
        loss = torch.abs(pred-gt)
    elif type == 'L2':
        loss = (pred - gt)**2
    loss = torch.mean(weight * loss)
    return loss

def loss_fn(ret, gt, weight=1, type='L1'):
    loss = None
    if 'proj' in ret:
        proj = ret['proj']
        loss = loss_cal(proj, gt, weight, type)
    return loss

def prob_reg_loss(model, num_points, device):
    random_pnts = torch.rand(num_points, 3).to(torch.float32).to(device) #[num_points, 3]
    prob = model.get_probability(random_pnts) #[num_points, 1]
    prob_reg = torch.mean(prob)
    return prob_reg

def prob_entropy_loss(model, num_points, device):
    random_pnts = torch.rand(num_points, 3).to(torch.float32).to(device)
    if torch.any(torch.isnan(random_pnts)):
        print("FATAL: NaN for random_pnts !")
    if torch.any(torch.isinf(random_pnts)):
        print("FATAL: Inf for random_pnts !")
    prob = model.get_probability(random_pnts)
    if torch.any(torch.isnan(prob)):
        print("FATAL: NaN for prob !")
    if torch.any(torch.isinf(prob)):
        print("FATAL: Inf for prob !")

    epsilon = 1e-7
    prob_clamped = torch.clamp(prob, min=epsilon, max=1-epsilon)
    entropy_loss = F.binary_cross_entropy(input=prob_clamped, target=prob_clamped)
    
    if torch.any(torch.isnan(entropy_loss)):
        print("FATAL: NaN for entropy !")
    if torch.any(torch.isinf(entropy_loss)):
        print("FATAL: Inf for entropy !")
    return entropy_loss
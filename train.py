from models.model import *
from data.RealXdataset import RealXdataset
from util.train_args import parse_args
from RealXtrainer import RealXtrainer

if __name__ == '__main__':

    args, conf = parse_args()
    device = args.device
    data = RealXdataset(args, device)

    nerf_model = VPAL(conf['model']).to(device)

    trainer = RealXtrainer(nerf_model, data, args, conf, device) 
    trainer.start()
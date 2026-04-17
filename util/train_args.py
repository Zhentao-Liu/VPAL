import os
import argparse
from pyhocon import ConfigFactory, HOCONConverter

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--nviews", "-V", type=int, default=30, help="Number of selected views",)
    parser.add_argument("--name", "-n", type=str, default='DSA_train', help="experiment name")
    parser.add_argument("--counter", type=int, default=-1, help="which exp for current exp series")
    parser.add_argument("--output_path", type=str, default="output", help="output directory",)
    parser.add_argument("--epochs",type=int,default=100000,help="number of epochs to train",)
    parser.add_argument("--history_epoch", type=int, default=0, help="resume from the history net")
    parser.add_argument("--datadir", "-D", type=str, default=None, help="Dataset directory")
    parser.add_argument("--geotype", type=str, default='vec', help="vec | matrix")
    parser.add_argument("--conf", "-c", type=str, default=None, help='Config file')
    parser.add_argument("--model", type=str, default='VPAL', help="use which model to train")
    parser.add_argument("--preg", type=float, default=0.01, help="prob regularization weight")
    parser.add_argument("--pentropy", type=float, default=0.01, help="prob entropy regularization weight")
    parser.add_argument("--PG", type=int, default=1, help="whether to use progressive training")
    parser.add_argument("--TP", type=int, default=1, help="whether to use temporal perturbation")
    parser.add_argument("--TP_std", type=float, default=1, help="control the radius of gaussian perturbtion")
    parser.add_argument("--lr_p", type=float, default=7.5e-4, help="initial learning rate for probability field")
    parser.add_argument("--lr_s", type=float, default=7.5e-4, help="initial learning rate for static field")
    parser.add_argument("--lr_d", type=float, default=7.5e-4, help="initial learning rate for dynamic field")
    parser.add_argument("--device", type=str, default='cuda', help='compute device')
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument("--is_train", action="store_true", help="Training or visualization")
    parser.add_argument("--disable_2d", action="store_true", help="disable 2d projection rotation view during visualization")
    parser.add_argument("--disable_fixview", action="store_true", help="disable 2d projection fix view during visualization")
    parser.add_argument("--fixview", type=int, default=0, help="view index for fix view 2d rendering")
    parser.add_argument("--disable_fixtime", action="store_true", help="disable 2d projection fix time during visualization")
    parser.add_argument("--fixtime", type=int, default=0, help="view index for fix time 2d rendering")   
    parser.add_argument("--disable_3d", action="store_true", help="disable 3d reconstruction during visualization")
    parser.add_argument("--occacc", action="store_true", help=(
            "Enable occupancy grid acceleration to skip empty space based on vessel probability (applicable for VPAL and PD_field)."
            "Note: this may degrade performance, especially for VPAL."))
    parser.add_argument("--out_other", action="store_true", help="Whether to output other terms")

    parser.add_argument("--noise_aug", action="store_true", help="Enable noise-like augmentation on input projections")
    parser.add_argument("--I0", type=float, default=1e5, help="Noise strength for noise augmentation")
    parser.add_argument("--GSstd", type=float, default=10, help="Noise strength for noise augmentation")
    parser.add_argument("--SAD_aug", action="store_true", help="Enable source-to-object distance (SAD) perturbation")
    parser.add_argument("--SAD_aug_size", type=float, default=1.0, help="Magnitude of SAD perturbation")
    parser.add_argument("--angle_aug", action="store_true", help="Enable projection angle perturbation")
    parser.add_argument("--angle_aug_size", type=float, default=0.3, help="Magnitude of angular perturbation (in degrees)")

    args = parser.parse_args()

    if args.TP and args.TP_std == 0:
        raise ValueError("TP_std must be non-zero when TP is enabled.")
    if not args.TP and args.TP_std != 0:
        raise ValueError("TP_std must be zero when TP is disabled.")

    args.PG = bool(args.PG)
    args.TP = bool(args.TP)

    if args.counter != -1:
        args.name = args.name + '_' + str(args.counter)
    logs_path = os.path.join(args.output_path, args.name, 'logs')
    os.makedirs(logs_path, exist_ok=True)

    load_conf = args.conf
    conf = ConfigFactory.parse_file(load_conf)

    if args.model in ['VPAL', 'PD_field']:
        conf.put("loss.prob_reg_weight", args.preg)
        conf.put("loss.prob_entropy_weight", args.pentropy)
        conf.put("occgrid.enabled", args.occacc)
    if args.model in ['VPAL', 'SD_field', 'PD_field', 'D_field']:
        conf.put("model.coarse2fine.enabled", args.PG)
        conf.put("flow_consistency.enabled", args.TP)
        conf.put("flow_consistency.TP_std", args.TP_std)
    if args.model == 'S_field':
        conf.put("model.coarse2fine.enabled", args.PG)
    
    if args.model == 'VPAL': 
        conf.put("optim_nerf.params.lr_p", args.lr_p)
        conf.put("optim_nerf.params.lr_s", args.lr_s)
        conf.put("optim_nerf.params.lr_d", args.lr_d)
    elif args.model == 'SD_field':
        conf.put("optim_nerf.params.lr_s", args.lr_s)
        conf.put("optim_nerf.params.lr_d", args.lr_d)
    elif args.model == 'PD_field':
        conf.put("optim_nerf.params.lr_p", args.lr_p)
        conf.put("optim_nerf.params.lr_d", args.lr_d)
    elif args.model == 'D_field':
        conf.put("optim_nerf.params.lr_d", args.lr_d)
    elif args.model in ['NAF', 'S_field']:
        conf.put("optim_nerf.params.lr_s", args.lr_s)        

    conf_outpath = os.path.join(logs_path, 'train.conf')
    if args.is_train and not os.path.exists(conf_outpath):
        with open(conf_outpath,'w') as f:
            f.write(HOCONConverter.convert(conf, 'hocon'))

    exp_state_list = ['Exp name: ', args.name, '\n',
                      'Training or not: ', "yes" if args.is_train else "no", '\n',
                      'Resume: ', "yes" if args.resume else "no", '\n',
                      'Dataset: ', args.datadir, '\n',
                      'Config file: ', args.conf, '\n',
                      'Model: ', args.model, '\n',
                      'Input views: ', str(args.nviews), '\n',
                      'Visualize 2D rotation view projection: ', "yes" if not args.disable_2d else "no", '\n',
                      'Visualize 2D fix view projection: ', "yes" if not args.disable_fixview else "no", '\n',
                      'fix view index: ', str(args.fixview), '\n',
                      'Visualize 2D fix time projection: ', "yes" if not args.disable_fixtime else "no", '\n',
                      'fix time index: ', str(args.fixtime), '\n',
                      'Visualize 3D reconstruction: ', "yes" if not args.disable_3d else "no", '\n',]

    exp_state = ''.join(exp_state_list)
    print(exp_state)
    f_exp = open(logs_path + '/exp_state.txt', mode='a')
    f_exp.write(exp_state)
    f_exp.close()
    return args,conf
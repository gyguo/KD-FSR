# test model with fsr module
import argparse
import os
import sys
import torch
from lib.utils import Logger
from scripts.engin import AverageMeter, map_sklearn, creat_data_loader, str_gpus
from scripts.config import cfg_from_list, cfg_from_file
from scripts.config import config as cfg
from lib.utils import fix_random_seed
import torch.backends.cudnn as cudnn

from lib.model.fsr_kd_x2 import FSR_SHUFFLE_X2
from lib.model.fsr_kd_x4 import FSR_SHUFFLE_X4

root_dir = os.path.join(os.path.dirname(__file__), '..')

import numpy as np

def load_evaluate_model(cfg, state_dict_file):
    print('==> load state...')
    # use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.GPU_ID)
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"

    if cfg.DATA.LARGE_SIZE/cfg.DATA.SMALL_SIZE == 4:
        model = FSR_SHUFFLE_X4(arch=cfg.FSR.SMALL_ARCH, in_planes=cfg.FSR.IN_PLANES,
                                   out_planes=cfg.FSR.OUT_PLANES, num_labels=cfg.DATA.NUM_CLASSES,
                                   feat_trim=cfg.FSR.FEATURE_TRIM, groups=cfg.FSR.GROUPS)
    elif cfg.DATA.LARGE_SIZE/cfg.DATA.SMALL_SIZE == 2:
        model = FSR_SHUFFLE_X2(arch=cfg.FSR.SMALL_ARCH, in_planes=cfg.FSR.IN_PLANES,
                                   out_planes=cfg.FSR.OUT_PLANES, num_labels=cfg.DATA.NUM_CLASSES,
                                   feat_trim=cfg.FSR.FEATURE_TRIM, groups=cfg.FSR.GROUPS)

    model = torch.nn.DataParallel(model).to(device)
    state_dict = torch.load(state_dict_file)['state_dict']
    model.load_state_dict(state_dict)
    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    print('Preparing networks done!')

    return device, model, criterion

def main():
    # ### configuration
    parser = argparse.ArgumentParser(description='Classification model training')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    # fix sedd
    fix_random_seed(cfg.SEED)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # path
    ckpt_dir = cfg.FSR.TEST.STATE_DIR
    state_dict_file = os.path.join(ckpt_dir, 'ckpt_fsr1/model_best_fsr1.pth')
    log_file = os.path.join(ckpt_dir, 'Log_FSR_{}_{}.txt'.format(cfg.DATA.INPUT_SIZE, cfg.TIME))
    # start loging
    sys.stdout = Logger(log_file)
    print(cfg)

    cfg.TEST = cfg.FSR.TEST
    device, model, criterion = load_evaluate_model(cfg, state_dict_file)
    train_loader, val_loader = creat_data_loader(cfg, root_dir)

    map_base, map_fsr = validate_one_epoch_baseline(val_loader, model, device, criterion, 0,  cfg)
    print(('MAP_BASE: {}\tMAP_FSR: {}'.format(map_base, map_fsr)))


def validate_one_epoch_baseline(val_loader, model, device, criterion, epoch, cfg):
    maps_base = AverageMeter()
    maps_fsr = AverageMeter()

    results_fsr = []
    results_base = []
    labels = []
    with torch.no_grad():
        model.eval()
        for i, (input, target) in enumerate(val_loader):

            target = target.to(device)
            input = input.to(device)

            output_base, _, _, output_fsr, _, _ = model(input)
            loss = criterion(output_base, target)
            map_base = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(output_base).detach().cpu().numpy())
            map_fsr = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(output_fsr).detach().cpu().numpy())


            res_fsr = output_fsr.detach().cpu().numpy()
            res_base = output_base.detach().cpu().numpy()
            if i==0:
                results_fsr = res_fsr
                results_base = res_base
                labels = target.data.cpu().numpy()
            else:
                results_base = np.concatenate((results_base, res_base), axis=0)
                results_fsr = np.concatenate((results_fsr, res_fsr), axis=0)
                labels = np.concatenate((labels, target.data.cpu().numpy()), axis=0)

            # record loss and map
            maps_base.update(map_base, input.size(0))
            maps_fsr.update(map_fsr, input.size(0))

            if i % cfg.DISP_FREQ == 0:
                print(('Baseline Validation Epoch: [{0}][{1}/{2}]\t'
                       'MAP_BASE {map_base.val:.4f} ({map_base.avg:.4f})'
                       'MAP_FSR {map_fsr.val:.4f} ({map_fsr.avg:.4f})'.format(
                    epoch+1, i+1, len(val_loader), map_base=maps_base, map_fsr=maps_fsr)))

    from sklearn.metrics import average_precision_score
    print('Low-Resolution Branch')
    print(average_precision_score(labels, results_base, average="micro"))
    print('High-Resolution Branch')
    print(average_precision_score(labels, results_fsr, average="micro"))
    return maps_base.avg, maps_fsr.avg


if __name__ == "__main__":
    main()
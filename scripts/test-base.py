# test model without fsr module
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
from lib.model.resnet_fsr import resnet18, resnet34, resnet50, resnet101

root_dir = os.path.join(os.path.dirname(__file__), '..')

import numpy as np

def load_evaluate_model(cfg, state_dict_file):
    print('==> load state...')
    # use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.GPU_ID)
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"

    if cfg.BASE.ARCH == 'r101':
        model = resnet101(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.BASE.ARCH == 'r50':
        model = resnet50(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.BASE.ARCH == 'r34':
        model = resnet34(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.BASE.ARCH == 'r18':
        model = resnet18(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)

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
    ckpt_dir = cfg.BASE.TEST.STATE_DIR
    state_dict_file = os.path.join(ckpt_dir, 'ckpt_base_s/model_best_s.pth')
    log_file = os.path.join(ckpt_dir, 'Log_Baseline_{}_{}.txt'.format(cfg.DATA.INPUT_SIZE, cfg.TIME))
    # start loging
    sys.stdout = Logger(log_file)
    print(cfg)

    cfg.TEST = cfg.BASE.TEST
    device, model, criterion = load_evaluate_model(cfg, state_dict_file)
    train_loader, val_loader = creat_data_loader(cfg, root_dir)

    loss_val, map_val = validate_one_epoch_baseline(val_loader, model, device, criterion, 0,  cfg)


def validate_one_epoch_baseline(val_loader, model, device, criterion, epoch, cfg):
    losses = AverageMeter()
    maps = AverageMeter()

    results = []
    labels = []
    with torch.no_grad():
        model.eval()
        for i, (input, target) in enumerate(val_loader):

            target = target.to(device)
            input = input.to(device)

            output, _, _ = model(input)
            loss = criterion(output, target)
            map = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(output).detach().cpu().numpy())


            res = output.detach().cpu().numpy()
            if i==0:
                results = res
                labels = target.data.cpu().numpy()
            else:
                results = np.concatenate((results, res), axis=0)
                labels = np.concatenate((labels, target.data.cpu().numpy()), axis=0)

            # record loss and map
            losses.update(loss.item(), input.size(0))
            maps.update(map, input.size(0))

            if i % cfg.DISP_FREQ == 0:
                print(('Baseline Validation Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'MAP {map.val:.4f} ({map.avg:.4f})'.format(
                    epoch+1, i+1, len(val_loader), loss=losses, map=maps)))

    from sklearn.metrics import average_precision_score
    print(average_precision_score(labels, results, average="micro"))
    return losses.avg, maps.avg


if __name__ == "__main__":
    main()
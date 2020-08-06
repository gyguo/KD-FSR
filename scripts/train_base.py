'''
Train and test baseline
@ author Guangyu Guo
'''
import argparse
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from lib.utils import mkdir, Logger
from scripts.engin import creat_data_loader, adjust_learning_rate, \
    str_gpus, AverageMeter, map_sklearn
from scripts.config import cfg_from_list, cfg_from_file
from scripts.config import config as cfg

from lib.utils import fix_random_seed
import torch.backends.cudnn as cudnn

root_dir = os.path.join(os.path.dirname(__file__), '..')

from lib.model.resnet_fsr import resnet34, resnet50, resnet101
def creat_model_baseline(cfg):
    print('==> Preparing networks for baseline...')
    # use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.GPU_ID)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    if cfg.BASE.ARCH == 'r101':
        base_model = resnet101(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.BASE.ARCH == 'r50':
        base_model = resnet50(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.BASE.ARCH == 'r34':
        base_model = resnet34(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    base_optimizer = torch.optim.SGD([{'params': base_model.parameters(), 'lr_mult': 1}],
                                     lr=cfg.BASE.SOLVER.START_LR,
                                     momentum=cfg.BASE.SOLVER.MUMENTUM,
                                     weight_decay=cfg.BASE.SOLVER.WEIGHT_DECAY)
    base_model = torch.nn.DataParallel(base_model).to(device)
    # loss
    cls_criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    print('Preparing networks done!')
    return device, base_model, base_optimizer, cls_criterion

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

    # create checkpoint directory
    save_dir = os.path.join('../ckpt', cfg.DATASET, '{}_{}_SIZE{}_SEED{}_{}'.format(
        cfg.MODEL.TYPE, cfg.BASE.ARCH, cfg.DATA.INPUT_SIZE, cfg.SEED, cfg.TIME))
    base_log_dir = os.path.join(save_dir, 'log_base'); mkdir(base_log_dir)
    base_ckpt_dir = os.path.join(save_dir, 'ckpt_base'); mkdir(base_ckpt_dir)
    log_file = os.path.join(save_dir, 'Log_'+cfg.TIME+'.txt')
    # start loging
    sys.stdout = Logger(log_file)
    print(cfg)

    # ######################### baseline ############################################
    # create baseline model and data loader
    cfg.TRAIN = cfg.BASE.TRAIN
    cfg.TEST = cfg.BASE.TEST
    cfg.SOLVER = cfg.BASE.SOLVER
    device, base_model, base_optimizer, cls_criterion = creat_model_baseline(cfg)

    base_train_loader, base_val_loader = creat_data_loader(cfg, root_dir)

    # train baseline
    base_best_map = 0
    base_update_train_step = 0
    base_update_val_step = 0
    base_writer = SummaryWriter(base_log_dir)
    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        adjust_learning_rate(base_optimizer, epoch, cfg)
        base_update_train_step, loss_train, map_train = \
            train_one_epoch_baseline(base_train_loader, base_model, device, cls_criterion,
                                     base_optimizer, epoch, base_writer, cfg, base_update_train_step)
        base_update_val_step, loss_val, map_val = \
            validate_one_epoch_baseline(base_val_loader, base_model, device, cls_criterion,
                                        epoch, base_writer, cfg, base_update_val_step)
        # log
        base_writer.add_scalar('Loss_L_epoch/train', loss_train, epoch+1)
        base_writer.add_scalar('Map_L_epoch/train', map_train, epoch+1)
        base_writer.add_scalar('Loss_L_epoch/test', loss_val, epoch+1)
        base_writer.add_scalar('Map_L_epoch/test', map_val, epoch+1)

        # save
        is_best = map_val > base_best_map
        base_best_map = max(map_val, base_best_map)
        if is_best:
            torch.save({
                'state_dict': base_model.state_dict(),
                'best_map': base_best_map
            }, os.path.join(base_ckpt_dir, 'model_best.pth'))
        print(('Baseline Epoch: {}, BEST_MAP: {}'
               .format(epoch+1, base_best_map)))


def train_one_epoch_baseline(train_loader, model, device, criterion, optimizer, epoch,
                    writer, cfg, update_train_step):
    batch_multiplier = cfg.TRAIN.BATCH_MULTIPLY
    losses = AverageMeter()
    maps = AverageMeter()

    model.train()
    count = 0
    for i, (input, target) in enumerate(train_loader):
        if count == 0:
            optimizer.step()
            optimizer.zero_grad()
            count = batch_multiplier

        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)
        output, _, feat = model(input)

        loss = criterion(output, target)
        loss.backward()
        count -= 1

        # record loss and map
        map = map_sklearn(target.detach().cpu().numpy(),
                          torch.sigmoid(output).detach().cpu().numpy())
        losses.update(loss.item(), input.size(0))
        maps.update(map, input.size(0))
        writer.add_scalar('Loss_iter/train', loss.item(), update_train_step)
        writer.add_scalar('Map_iter/train', map, update_train_step)

        if i % cfg.DISP_FREQ == 0:
            print(('Baseline Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'MAP {map.val:.4f} ({map.avg:.4f})'.format(
                epoch+1, i+1, len(train_loader), loss=losses, map=maps,
                lr=optimizer.param_groups[-1]['lr'])))

    return update_train_step, losses.avg, maps.avg


def validate_one_epoch_baseline(val_loader, model, device, criterion, epoch, writer, cfg, update_val_step):
    losses = AverageMeter()
    maps = AverageMeter()
    with torch.no_grad():
        model.eval()
        for i, (input, target) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input = input.to(device)

            output, _, feat = model(input)
            loss = criterion(output, target)
            map = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(output).detach().cpu().numpy())

            # record loss and map
            losses.update(loss.item(), input.size(0))
            maps.update(map, input.size(0))
            writer.add_scalar('Loss_iter/test', loss.item(), update_val_step)
            writer.add_scalar('Map_iter/test', map, update_val_step)

            if i % cfg.DISP_FREQ == 0:
                print(('Baseline Validation Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'MAP {map.val:.4f} ({map.avg:.4f})'.format(
                    epoch+1, i+1, len(val_loader), loss=losses, map=maps)))

        return update_val_step, losses.avg, maps.avg

if __name__ == "__main__":
    main()

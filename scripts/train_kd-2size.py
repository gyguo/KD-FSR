'''
one stage, train kd with 2 size
@ author Guangyu Guo
'''
import argparse
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from lib.utils import mkdir, Logger
from scripts.engin import adjust_learning_rate, creat_data_loader, \
    AverageMeter, map_sklearn, str_gpus, DistillKL
from scripts.config import cfg_from_list, cfg_from_file
from scripts.config import config as cfg

from lib.utils import fix_random_seed
import torch.backends.cudnn as cudnn
# close warning
import warnings
warnings.filterwarnings("ignore")

root_dir = os.path.join(os.path.dirname(__file__), '..')


from lib.model.resnet_fsr_cam import resnet18, resnet34, resnet50, resnet101


def creat_model_student(cfg):
    print('==> Preparing networks for knowledge distilllation...')
    # use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.GPU_ID)
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"

    if cfg.KD.ARCH_S == 'r101':
        model_s = resnet101(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.KD.ARCH_S == 'r50':
        model_s = resnet50(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.KD.ARCH_S == 'r34':
        model_s = resnet34(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.KD.ARCH_S == 'r18':
        model_s = resnet18(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    optimizer_s = torch.optim.SGD([{'params': model_s.parameters(), 'lr_mult': 1}],
                                     lr=cfg.KD.SOLVER.START_LR,
                                     momentum=cfg.KD.SOLVER.MUMENTUM,
                                     weight_decay=cfg.KD.SOLVER.WEIGHT_DECAY)
    model_s = torch.nn.DataParallel(model_s).to(device)

    # loss
    cls_criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    kd_criterion = DistillKL(cfg.KD.TEMP).to(device)
    print('Preparing networks done!')
    return device, model_s, optimizer_s, cls_criterion, kd_criterion


def load_model_teacher(cfg, device, root):
    print('==> load pre-trained teacher models...')
    if cfg.KD.ARCH_T == 'r101':
        model_t = resnet101(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.KD.ARCH_T == 'r50':
        model_t = resnet50(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.KD.ARCH_T == 'r34':
        model_t = resnet34(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    elif cfg.KD.ARCH_T == 'r18':
        model_t = resnet18(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    model_t = torch.nn.DataParallel(model_t).to(device)
    state_dict_t = torch.load(os.path.join(root, cfg.KD.MODELDICT_T))['state_dict']
    model_t.load_state_dict(state_dict_t)
    print('done!')
    return model_t

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
    save_dir = os.path.join('../ckpt', cfg.DATASET,
                            'KD-2SIZE_ARCH{}-{}_SIZE{}-{}_SEED{}_{}'.format(
                                cfg.KD.ARCH_T, cfg.KD.ARCH_S, cfg.DATA.LARGE_SIZE,
                                cfg.DATA.SMALL_SIZE, cfg.SEED, cfg.TIME))
    log_dir = os.path.join(save_dir, 'log'); mkdir(log_dir)
    ckpt_dir_t = os.path.join(save_dir, 'ckpt_base_t'); mkdir(ckpt_dir_t)
    ckpt_dir_s = os.path.join(save_dir, 'ckpt_base_s'); mkdir(ckpt_dir_s)
    log_file = os.path.join(save_dir, 'Log_'+cfg.TIME+'.txt')
    # start loging
    sys.stdout = Logger(log_file)
    print(cfg)

    # create baseline model and data loader
    train_loader, val_loader = creat_data_loader(cfg, root_dir)
    device, model_s, optimizer_s, cls_criterion, kd_criterion = creat_model_student(cfg)
    model_t = load_model_teacher(cfg, device, root_dir)

    # train
    best_map_s = 0
    update_train_step = 0
    update_val_step = 0
    writer = SummaryWriter(log_dir)
    cfg.SOLVER = cfg.KD.SOLVER
    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        adjust_learning_rate(optimizer_s, epoch, cfg)
        update_train_step,  map_t_train, loss_cls_s_train, \
        map_s_train, loss_kd_train = \
            train_one_epoch_fsr(train_loader, model_t, model_s, device, kd_criterion,
                                cls_criterion, optimizer_s, epoch, writer, cfg, update_train_step)

        update_val_step,  map_t_val,  loss_cls_s_val, map_s_val, loss_kd_val = \
            val_one_epoch_fsr(val_loader, model_t, model_s, device, kd_criterion, cls_criterion,
                              epoch, writer, cfg, update_val_step)
        print('map_t {}\tmap_s {}'.format(map_t_val, map_s_val))

        writer.add_scalar('loss_cls_s/train', loss_cls_s_train, epoch+1)
        writer.add_scalar('loss_kd/train', loss_kd_train, epoch+1)
        writer.add_scalar('map_t/train', map_t_train, epoch+1)
        writer.add_scalar('map_s/train', map_s_train, epoch+1)
        writer.add_scalar('loss_cls_s/val', loss_cls_s_val, epoch+1)
        writer.add_scalar('loss_kd/val', loss_kd_val, epoch+1)
        writer.add_scalar('map_t/val', map_t_val, epoch+1)
        writer.add_scalar('map_s/val', map_s_val, epoch+1)

        # print and save
        print(('Epoch: {}, MAP_T: {}'.format(epoch+1, map_t_val)))
        is_best_model_s = map_s_val > best_map_s
        best_map_s = max(map_s_val, best_map_s)
        if is_best_model_s:
            torch.save({
                'epoch': epoch,
                'state_dict': model_s.state_dict(),
                'best_map': best_map_s
            }, os.path.join(ckpt_dir_s, 'model_best_s.pth'))
        print(('Epoch: {}, BEST_MAP_S: {}'.format(epoch + 1, best_map_s)))


def train_one_epoch_fsr(train_loader, model_t, model_s, device, kd_criterion,
                        cls_criterion, optimizer_s, epoch, writer, cfg, update_train_step):

    batch_multiplier = cfg.TRAIN.BATCH_MULTIPLY
    loss_weights = cfg.FSR.TRAIN.LOSS_WEIGHTS

    maps_t = AverageMeter()
    losses_cls_s = AverageMeter()
    maps_s = AverageMeter()
    losses_kd = AverageMeter()

    model_t.eval()
    model_s.train()
    count = 0
    for i, (input_l, input_s, target) in enumerate(train_loader):
        if count == 0:
            optimizer_s.step()
            optimizer_s.zero_grad()
            count = batch_multiplier

        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input_l = input_l.to(device)
        input_s = input_s.to(device)

        # train teacher model
        with torch.no_grad():
            pret_t, _, _, _ = model_t(input_l)
        map_t = map_sklearn(target.detach().cpu().numpy(),
                          torch.sigmoid(pret_t).detach().cpu().numpy())
        maps_t.update(map_t, input_s.size(0))
        if i % cfg.DISP_FREQ == 0:
            print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                   'MAP_T {map_t.val:.4f} ({map_t.avg:.4f})\t'.format(
                epoch+1, i+1, len(train_loader),
                lr=optimizer_s.param_groups[0]['lr'], map_t=maps_t)))

        # train student model
        pret_s, _, _, _ = model_s(input_s)
        loss_cls_s = cls_criterion(pret_s, target)
        loss_kd = kd_criterion(pret_s, pret_t.detach())
        loss_s = (1-cfg.KD.ALPHA)*loss_cls_s + 2*cfg.KD.ALPHA*loss_kd
        loss_s.backward()

        # log student
        losses_cls_s.update(loss_cls_s.item(), input_s.size(0))
        map_s = map_sklearn(target.detach().cpu().numpy(),
                          torch.sigmoid(pret_s).detach().cpu().numpy())
        maps_s.update(map_s, input_s.size(0))
        losses_kd.update(loss_kd, input_s.size(0))

        if i % cfg.DISP_FREQ == 0:
            print(('LOSS_CLS_S {loss_cls_s.val:.4f} ({loss_cls_s.avg:.4f})\t'
                   'MAP_S {map_s.val:.4f} ({map_s.avg:.4f})\n'
                   'LOSS_KD {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t'.format(
                loss_cls_s=losses_cls_s, map_s=maps_s, loss_kd=losses_kd)))
        count -= 1

        writer.add_scalar('LOSS_CLS_S_ITER/train', loss_cls_s.item(), update_train_step)
        writer.add_scalar('LOSS_KD_ITER/train', loss_kd.item(), update_train_step)
        writer.add_scalar('MAP_ITER/train/map_t', map_t, update_train_step)
        writer.add_scalar('MAP_ITER/train/map_s', map_s, update_train_step)

    return update_train_step, maps_t.avg, \
           losses_cls_s.avg, maps_s.avg, losses_kd.avg


def val_one_epoch_fsr(val_loader, model_t, model_s, device, kd_criterion, cls_criterion,
                         epoch, writer, cfg, update_val_step):

    maps_t = AverageMeter()
    losses_cls_s = AverageMeter()
    maps_s = AverageMeter()
    losses_kd = AverageMeter()

    with torch.no_grad():
        model_t.eval()
        model_s.eval()
        for i, (input_l, input_s, target) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input_l = input_l.to(device)
            input_s = input_s.to(device)

            # teacher model
            pret_t, _, _, _ = model_t(input_l)
            map_t = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(pret_t).detach().cpu().numpy())
            maps_t.update(map_t, input_s.size(0))
            if i % cfg.DISP_FREQ == 0:
                print(('Validation Epoch: [{0}][{1}/{2}]\t'
                        'MAP_T {map_t.val:.4f} ({map_t.avg:.4f})\t'.format(
                    epoch+1, i+1, len(val_loader), map_t=maps_t)))

            # student model
            pret_s, _, _, _ = model_s(input_s)
            loss_cls_s = cls_criterion(pret_s, target)
            loss_kd = kd_criterion(pret_s, pret_t.detach())

            # log student
            losses_cls_s.update(loss_cls_s.item(), input_s.size(0))
            map_s = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(pret_s).detach().cpu().numpy())
            maps_s.update(map_s, input_s.size(0))
            losses_kd.update(loss_kd, input_s.size(0))

            if i % cfg.DISP_FREQ == 0:
                print(('LOSS_CLS_S {loss_cls_s.val:.4f} ({loss_cls_s.avg:.4f})\t'
                       'MAP_S {map_s.val:.4f} ({map_s.avg:.4f})\n'
                       'LOSS_KD {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t'.format(
                    loss_cls_s=losses_cls_s, map_s=maps_s, loss_kd=losses_kd)))

            writer.add_scalar('LOSS_CLS_S_ITER/val', loss_cls_s.item(), update_val_step)
            writer.add_scalar('LOSS_KD_ITER/val', loss_kd.item(), update_val_step)
            writer.add_scalar('MAP_ITER/val/map_t', map_t, update_val_step)
            writer.add_scalar('MAP_ITER/val/map_s', map_s, update_val_step)

    return update_val_step, maps_t.avg, \
           losses_cls_s.avg, maps_s.avg, losses_kd.avg


if __name__ == "__main__":
    main()
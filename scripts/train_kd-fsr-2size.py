'''
two stage kd-fsr
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
from lib.model.fsr_kd_x2 import FSR_SHUFFLE_X2
from lib.model.fsr_kd_x4 import FSR_SHUFFLE_X4

def creat_model_student_fsr(cfg):
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

    if cfg.DATA.LARGE_SIZE/cfg.DATA.SMALL_SIZE == 4:
        model_fsr = FSR_SHUFFLE_X4(arch=cfg.FSR.SMALL_ARCH, in_planes=cfg.FSR.IN_PLANES,
                                   out_planes=cfg.FSR.OUT_PLANES, num_labels=cfg.DATA.NUM_CLASSES,
                                   feat_trim=cfg.FSR.FEATURE_TRIM, groups=cfg.FSR.GROUPS)
    elif cfg.DATA.LARGE_SIZE/cfg.DATA.SMALL_SIZE == 2:
        model_fsr = FSR_SHUFFLE_X2(arch=cfg.FSR.SMALL_ARCH, in_planes=cfg.FSR.IN_PLANES,
                                   out_planes=cfg.FSR.OUT_PLANES, num_labels=cfg.DATA.NUM_CLASSES,
                                   feat_trim=cfg.FSR.FEATURE_TRIM, groups=cfg.FSR.GROUPS)
    optimizer_fsr = torch.optim.SGD([{'params': model_fsr.parameters(), 'lr_mult': 1}],
                                    lr=cfg.FSR.SOLVER.START_LR,
                                    momentum=cfg.FSR.SOLVER.MUMENTUM,
                                    weight_decay=cfg.FSR.SOLVER.WEIGHT_DECAY)
    model_fsr = torch.nn.DataParallel(model_fsr).to(device)

    # loss
    cls_criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    kd_criterion = DistillKL(cfg.KD.TEMP).to(device)
    fsr_criterion = torch.nn.MSELoss().to(device)
    print('Preparing networks done!')
    return device, model_s, optimizer_s, cls_criterion, kd_criterion, model_fsr, optimizer_fsr, fsr_criterion


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
                            'KD-FSR-2SIZE_ARCH{}-{}_SIZE{}-{}_SEED{}_{}'.format(
                                cfg.KD.ARCH_T, cfg.KD.ARCH_S, cfg.DATA.LARGE_SIZE,
                                cfg.DATA.SMALL_SIZE, cfg.SEED, cfg.TIME))
    log_dir = os.path.join(save_dir, 'log'); mkdir(log_dir)
    ckpt_dir_t = os.path.join(save_dir, 'ckpt_base_t'); mkdir(ckpt_dir_t)
    ckpt_dir_s = os.path.join(save_dir, 'ckpt_base_s'); mkdir(ckpt_dir_s)
    ckpt_dir_fsr = os.path.join(save_dir, 'ckpt_fsr'); mkdir(ckpt_dir_fsr)
    log_file = os.path.join(save_dir, 'Log_'+cfg.TIME+'.txt')
    # start loging
    sys.stdout = Logger(log_file)
    print(cfg)

    # create baseline model and data loader
    train_loader, val_loader = creat_data_loader(cfg, root_dir)
    device, model_s, optimizer_s, cls_criterion, kd_criterion, \
    model_fsr, optimizer_fsr, fsr_criterion = creat_model_student_fsr(cfg)
    model_t = load_model_teacher(cfg, device, root_dir)

    # train
    best_map_s = 0
    best_map_fsr = 0
    best_base_map = 0
    update_train_step = 0
    update_val_step = 0
    writer = SummaryWriter(log_dir)
    cfg.SOLVER = cfg.KD.SOLVER
    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        adjust_learning_rate(optimizer_s, epoch, cfg)
        update_train_step,  map_t_train, loss_cls_s_train, map_s_train, loss_kd_train,\
        base_loss_cls_train, base_map_train, fsr_loss_cls_train, fsr_loss_fsr_train, fsr_map_train = \
            train_one_epoch_kd_fsr(train_loader, model_t, model_s, model_fsr, device,
                                   kd_criterion, cls_criterion, fsr_criterion, optimizer_s,
                                   optimizer_fsr, epoch, writer, cfg, update_train_step)

        update_val_step,  map_t_val,  loss_cls_s_val, map_s_val, loss_kd_val, \
        base_loss_cls_val, base_map_val, fsr_loss_cls_val, fsr_loss_fsr_val, fsr_map_val = \
            val_one_epoch_kd_fsr(val_loader, model_t, model_s, model_fsr, device,
                                 kd_criterion, cls_criterion, fsr_criterion,
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

        writer.add_scalar('base_loss_cls/train', base_loss_cls_train, epoch+1)
        writer.add_scalar('base_map/train', base_map_train, epoch+1)
        writer.add_scalar('fsr_loss_cls/train', fsr_loss_cls_train, epoch+1)
        writer.add_scalar('fsr_loss_fsr/train', fsr_loss_fsr_train, epoch+1)
        writer.add_scalar('fsr_map/train', fsr_map_train, epoch+1)
        writer.add_scalar('base_loss_cls/val', base_loss_cls_val, epoch+1)
        writer.add_scalar('base_map/val', base_map_val, epoch+1)
        writer.add_scalar('fsr_loss_cls/val', fsr_loss_cls_val, epoch+1)
        writer.add_scalar('fsr_loss_fsr/val', fsr_loss_fsr_val, epoch+1)
        writer.add_scalar('fsr_map/val', fsr_map_val, epoch+1)

        # print and save
        print(('Epoch: {}, MAP_T: {}'.format(epoch+1, map_t_val)))
        is_best_s = map_s_val > best_map_s
        best_map_s = max(map_s_val, best_map_s)
        if is_best_s:
            torch.save({
                'epoch': epoch,
                'state_dict': model_s.state_dict(),
                'best_map': best_map_s
            }, os.path.join(ckpt_dir_s, 'model_best_s.pth'))
        print(('Epoch: {}, BEST_MAP_S: {}'.format(epoch + 1, best_map_s)))

        is_best_fsr = fsr_map_val > best_map_fsr
        best_map_fsr = max(fsr_map_val, best_map_fsr)
        if is_best_fsr:
            torch.save({
                'epoch': epoch,
                'state_dict': model_fsr.state_dict(),
                'best_map': best_map_fsr
            }, os.path.join(ckpt_dir_fsr, 'model_best_fsr.pth'))
        print(('Epoch: {}, BEST_MAP_FSR: {}'.format(epoch + 1, best_map_fsr)))

        is_best_base = base_map_val > best_base_map
        best_base_map = max(base_map_val, best_base_map)
        if is_best_base:
            torch.save({
                'epoch': epoch,
                'state_dict': model_fsr.state_dict(),
                'best_map': best_base_map
            }, os.path.join(ckpt_dir_fsr, 'model_best_base.pth'))
        print(('Epoch: {}, BEST_BASE_MAP: {}'.format(epoch + 1, best_base_map)))


def train_one_epoch_kd_fsr(train_loader, model_t, model_s, model_fsr, device, kd_criterion, cls_criterion,
                           fsr_criterion, optimizer_s, optimizer_fsr, epoch, writer, cfg, update_train_step):

    batch_multiplier = cfg.TRAIN.BATCH_MULTIPLY
    loss_weights = cfg.FSR.TRAIN.LOSS_WEIGHTS

    maps_t = AverageMeter()
    losses_cls_s = AverageMeter()
    maps_s = AverageMeter()
    losses_kd = AverageMeter()

    base_losses_cls = AverageMeter()
    base_maps = AverageMeter()
    fsr_losses_cls = AverageMeter()
    fsr_maps = AverageMeter()
    fsr_losses_fsr = AverageMeter()

    model_t.eval()
    model_s.train()
    model_fsr.train()
    count = 0
    for i, (input_l, input_s, target) in enumerate(train_loader):
        if count == 0:
            optimizer_s.step()
            optimizer_s.zero_grad()
            optimizer_fsr.step()
            optimizer_fsr.zero_grad()
            count = batch_multiplier

        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input_l = input_l.to(device)
        input_s = input_s.to(device)

        # ## teacher model
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

        # ## train student model
        pret_s, _, f_feat_s, _ = model_s(input_l)
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

        # ## fsr student model
        pret_base, _, f_feat_base, pret_fsr, feat_fsr, _ = model_fsr(input_s)
        base_loss_cls = cls_criterion(pret_base, target)
        fsr_loss_fsr = fsr_criterion(feat_fsr, f_feat_s.detach())
        fsr_loss_cls = cls_criterion(pret_fsr, target)
        fsr_loss = base_loss_cls + fsr_loss_cls*loss_weights[0] + fsr_loss_fsr*loss_weights[1]
        fsr_loss.backward()
        # log fsr
        base_losses_cls.update(base_loss_cls.item(), input_s.size(0))
        base_map = map_sklearn(target.detach().cpu().numpy(),
                          torch.sigmoid(pret_base).detach().cpu().numpy())
        base_maps.update(base_map, input_s.size(0))
        fsr_losses_cls.update(fsr_loss_cls.item(), input_s.size(0))
        fsr_map = map_sklearn(target.detach().cpu().numpy(),
                          torch.sigmoid(pret_fsr).detach().cpu().numpy())
        fsr_maps.update(fsr_map, input_s.size(0))
        fsr_losses_fsr.update(fsr_loss_fsr.item(), input_s.size(0))
        if i % cfg.DISP_FREQ == 0:
            print(('BASE_LOSS_CLS {base_loss_cls.val:.4f} ({base_loss_cls.avg:.4f})\t'
                   'BASE_MAP {base_map.val:.4f} ({base_map.avg:.4f})\n'
                   'FSR_LOSS_CLS {fsr_loss_cls.val:.4f} ({fsr_loss_cls.avg:.4f})\t'
                   'FSR_MAP {fsr_map.val:.4f} ({fsr_map.avg:.4f})\t'
                   'FSR_LOSS_FSR {fsr_loss_fsr.val:.4f} ({fsr_loss_fsr.avg:.4f})' .format(
                base_loss_cls=base_losses_cls, base_map=base_maps,
                fsr_loss_cls=fsr_losses_cls, fsr_map=fsr_maps, fsr_loss_fsr=fsr_losses_fsr)))
        count -= 1

        writer.add_scalar('LOSS_CLS_S_ITER/train', loss_cls_s.item(), update_train_step)
        writer.add_scalar('LOSS_KD_ITER/train', loss_kd.item(), update_train_step)
        writer.add_scalar('MAP_ITER/train/map_t', map_t, update_train_step)
        writer.add_scalar('MAP_ITER/train/map_s', map_s, update_train_step)

        writer.add_scalar('BASE_LOSS_CLS_ITER/train', base_loss_cls.item(), update_train_step)
        writer.add_scalar('FSR_LOSS_CLS_ITER/train', fsr_loss_cls.item(), update_train_step)
        writer.add_scalar('FSR_LOSS_FSR_ITER/train', fsr_loss_fsr.item(), update_train_step)
        writer.add_scalar('MAP_ITER/train/base_map', base_map, update_train_step)
        writer.add_scalar('MAP_ITER/train/fsr_map', fsr_map, update_train_step)

    return update_train_step, maps_t.avg, losses_cls_s.avg, maps_s.avg, losses_kd.avg, \
           base_losses_cls.avg, base_maps.avg, fsr_losses_cls.avg, fsr_losses_fsr.avg, fsr_maps.avg


def val_one_epoch_kd_fsr(val_loader, model_t, model_s, model_fsr, device, kd_criterion,
                         cls_criterion, fsr_criterion, epoch, writer, cfg, update_val_step):

    maps_t = AverageMeter()
    losses_cls_s = AverageMeter()
    maps_s = AverageMeter()
    losses_kd = AverageMeter()

    base_losses_cls = AverageMeter()
    base_maps = AverageMeter()
    fsr_losses_cls = AverageMeter()
    fsr_maps = AverageMeter()
    fsr_losses_fsr = AverageMeter()

    with torch.no_grad():
        model_t.eval()
        model_s.eval()
        model_fsr.eval()
        for i, (input_l, input_s, target) in enumerate(val_loader):
            # update iteration steps
            update_val_step += 1

            target = target.to(device)
            input_l = input_l.to(device)
            input_s = input_s.to(device)

            # ## teacher model
            pret_t, _, _, _ = model_t(input_l)
            map_t = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(pret_t).detach().cpu().numpy())
            maps_t.update(map_t, input_s.size(0))
            if i % cfg.DISP_FREQ == 0:
                print(('Validation Epoch: [{0}][{1}/{2}]\t'
                        'MAP_T {map_t.val:.4f} ({map_t.avg:.4f})\t'.format(
                    epoch+1, i+1, len(val_loader), map_t=maps_t)))

            # ## student model
            pret_s, _, f_feat_s, _ = model_s(input_l)
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

            # ## fsr student model
            pret_base, _, f_feat_base, pret_fsr, feat_fsr, _ = model_fsr(input_s)
            base_loss_cls = cls_criterion(pret_base, target)
            fsr_loss_fsr = fsr_criterion(feat_fsr, f_feat_s.detach())
            fsr_loss_cls = cls_criterion(pret_fsr, target)
            # log fsr
            base_losses_cls.update(base_loss_cls.item(), input_s.size(0))
            base_map = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(pret_base).detach().cpu().numpy())
            base_maps.update(base_map, input_s.size(0))
            fsr_losses_cls.update(fsr_loss_cls.item(), input_s.size(0))
            fsr_map = map_sklearn(target.detach().cpu().numpy(),
                              torch.sigmoid(pret_fsr).detach().cpu().numpy())
            fsr_maps.update(fsr_map, input_s.size(0))
            fsr_losses_fsr.update(fsr_loss_fsr.item(), input_s.size(0))
            if i % cfg.DISP_FREQ == 0:
                print(('BASE_LOSS_CLS {base_loss_cls.val:.4f} ({base_loss_cls.avg:.4f})\t'
                       'BASE_MAP {base_map.val:.4f} ({base_map.avg:.4f})\n'
                       'FSR_LOSS_CLS {fsr_loss_cls.val:.4f} ({fsr_loss_cls.avg:.4f})\t'
                       'FSR_MAP {fsr_map.val:.4f} ({fsr_map.avg:.4f})\t'
                       'FSR_LOSS_FSR {fsr_loss_fsr.val:.4f} ({fsr_loss_fsr.avg:.4f})' .format(
                    base_loss_cls=base_losses_cls, base_map=base_maps,
                    fsr_loss_cls=fsr_losses_cls, fsr_map=fsr_maps, fsr_loss_fsr=fsr_losses_fsr)))

            writer.add_scalar('LOSS_CLS_S_ITER/val', loss_cls_s.item(), update_val_step)
            writer.add_scalar('LOSS_KD_ITER/val', loss_kd.item(), update_val_step)
            writer.add_scalar('MAP_ITER/val/map_t', map_t, update_val_step)
            writer.add_scalar('MAP_ITER/val/map_s', map_s, update_val_step)

            writer.add_scalar('BASE_LOSS_CLS_ITER/val', base_loss_cls.item(), update_val_step)
            writer.add_scalar('FSR_LOSS_CLS_ITER/val', fsr_loss_cls.item(), update_val_step)
            writer.add_scalar('FSR_LOSS_FSR_ITER/val', fsr_loss_fsr.item(), update_val_step)
            writer.add_scalar('MAP_ITER/val/base_map', base_map, update_val_step)
            writer.add_scalar('MAP_ITER/val/fsr_map', fsr_map, update_val_step)

    return update_val_step, maps_t.avg, losses_cls_s.avg, maps_s.avg, losses_kd.avg, \
           base_losses_cls.avg, base_maps.avg, fsr_losses_cls.avg, fsr_losses_fsr.avg, fsr_maps.avg


if __name__ == "__main__":
    main()
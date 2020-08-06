import os
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from dataset.voc import VOCClassification
from dataset.coco import COCO
from model.resnet_fsr import resnet101
import torch.nn.functional as F


class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, temp):
        super(DistillKL, self).__init__()
        self.temp = temp

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.temp, dim=1)
        p_t = F.softmax(y_t/self.temp, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.temp**2) / y_s.shape[0]
        return loss


class similarity_preserving_loss(torch.nn.Module):
    def __init__(self,):
        super(similarity_preserving_loss, self).__init__()

    def forward(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

class sp_loss(torch.nn.Module):
    def __init__(self):
        super(sp_loss, self).__init__()
        self.sp_criterion = similarity_preserving_loss()

    def forward(self, pred_feats, target_feats):
        loss = 0
        for pred_feat, target_feat in zip(pred_feats, target_feats):
            loss = loss + self.sp_criterion(pred_feat, target_feat.detach())
        return loss


def map_sklearn(labels, results):
    map = average_precision_score(labels, results, average="micro")
    return map


def str_gpus(ids):
    str_ids = ''
    for id in ids:
        str_ids =  str_ids + str(id)
        str_ids =  str_ids + ','

    return str_ids


def creat_model_baseline(cfg):
    print('==> Preparing networks for baseline...')
    # use gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str_gpus(cfg.GPU_ID)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    base_model = resnet101(pretrained=True, num_labels=cfg.DATA.NUM_CLASSES)
    ignored_params = list(map(id, base_model.fc_all.parameters()))
    trained_params = filter(lambda p: id(p) not in ignored_params, base_model.parameters())
    base_optimizer = torch.optim.SGD([{'params': trained_params, 'lr_mult': 1},
                                 {'params': base_model.fc_all.parameters(), 'lr_mult': 1}],
                                lr=cfg.BASE.SOLVER.START_LR,
                                momentum=cfg.BASE.SOLVER.MUMENTUM,
                                weight_decay=cfg.BASE.SOLVER.WEIGHT_DECAY)
    base_model = torch.nn.DataParallel(base_model).to(device)
    # loss
    cls_criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    print('Preparing networks done!')
    return device, base_model, base_optimizer, cls_criterion


def creat_data_loader(cfg, root_dir):
    print('==> Preparing data...')
    if cfg.DATASET == 'VOC2007':
        train_loader = torch.utils.data.DataLoader(
            VOCClassification(root=root_dir, cfg=cfg, split='trainval', is_train=True),
            batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            VOCClassification(root=root_dir, cfg=cfg, split='test', is_train=False),
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    elif cfg.DATASET == 'COCO':
        train_loader = torch.utils.data.DataLoader(
            COCO(root=root_dir, cfg=cfg, split='train', is_train=True),
            batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            COCO(root=root_dir, cfg=cfg, split='val', is_train=False),
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print('done!')
    return train_loader, val_loader


def adjust_learning_rate(optimizer, epoch, cfg):
    """"Sets the learning rate to the initial LR decayed by lr_factor"""
    lr_decay = cfg.SOLVER.LR_FACTOR**(sum(epoch >= np.array(cfg.SOLVER.LR_STEPS)))
    lr = cfg.SOLVER.START_LR* lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

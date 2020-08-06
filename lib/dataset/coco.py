import os
import json
import sys
import collections
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from dataset.utils import gen_base_transform, gen_fsr_transform


def gen_annotation_file(root, split):
    # load annotation
    anno_file = os.path.join(root, 'annotations', 'instances_{}2014.json'.format(split))
    anno_info = json.load(open(anno_file))
    annotations = anno_info['annotations']
    category = anno_info['categories']
    images = anno_info['images']

    list_file = os.path.join(root, '{}_anno.json'.format(split))

    img_id = {}
    annotations_id = {}
    category_id = {}
    for cat in category:
        category_id[cat['id']] = cat['name']
    cat2idx = categoty_to_idx(sorted(category_id.values()))
    for annotation in annotations:
        if annotation['image_id'] not in annotations_id:
            annotations_id[annotation['image_id']] = set()
        annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
    for img in images:
        if img['id'] not in annotations_id:
            continue
        if img['id'] not in img_id:
            img_id[img['id']] = {}
        img_id[img['id']]['file_name'] = img['file_name']
        img_id[img['id']]['labels'] = list(annotations_id[img['id']])
    anno_list = []
    for k, v in img_id.items():
        anno_list.append(v)
    json.dump(anno_list, open(list_file, 'w'))
    json.dump(cat2idx, open(os.path.join(root, 'category.json'), 'w'))


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO(data.Dataset):
    """
    Args:
        root (string): Root directory of the COCO Dataset.
        config: data configurations
        image_set (string, optional): Select the image_set to use, ``train``,  ``val``
        is_train(bool): train or evaluate
    """
    def __init__(self,
                 root,
                 cfg,
                 split='train',
                 is_train=True):
        self.num_classes = 80
        self.is_train = is_train
        self.split = split
        self.type = cfg.MODEL.TYPE
        coco_root = os.path.join(root, cfg.DATADIR)

        # load list
        list_file = os.path.join(coco_root, '{}_anno.json'.format(split))
        if not os.path.exists(list_file):
            gen_annotation_file(coco_root, split)
        self.image_list = json.load(open(list_file, 'r'))
        self.image_dir = os.path.join(coco_root, '{}2014'.format(split))
        self.cat2idx = json.load(open(os.path.join(coco_root, 'category.json'), 'r'))

        # data augmentation
        # data augmentation
        if cfg.MODEL.TYPE == 'BASELINE' or cfg.MODEL.TYPE == 'KD':
            self.train_transform, self.test_transform = gen_base_transform(cfg)
        elif cfg.MODEL.TYPE == 'FSR':
            self.train_transform_base, self.train_transform_l,\
            self.train_transform_s, self.test_transform_l,\
            self.test_transform_s = gen_fsr_transform(cfg)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        item = self.image_list[index]

        labels = sorted(item['labels'])
        target = torch.FloatTensor(self.num_classes).zero_()
        target[labels] = 1
        filename = item['file_name']
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        if self.type == 'BASELINE' or self.type == 'KD':
            if self.is_train:
                image = self.train_transform(image)
            else:
                image = self.test_transform(image)
            return image, target
        elif self.type == 'FSR':
            if self.is_train:
                image_base = self.train_transform_base(image)
                image_l = self.train_transform_l(image_base)
                image_s = self.train_transform_s(image_base)
            else:
                image_l = self.test_transform_l(image)
                image_s = self.test_transform_s(image)
            return image_l, image_s, target

    def __len__(self):
        return len(self.image_list)
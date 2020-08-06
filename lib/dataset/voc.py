import os
import sys
import collections
import numpy as np

import torch
import torch.utils.data as data
from dataset.utils import gen_base_transform, gen_fsr_transform

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from PIL import Image

DATASET_CLASS_DICT = {'aeroplane': 0, 'bicycle':1 , 'bird': 2, 'boat':3 , 'bottle': 4, 'bus':5, 'car': 6,
             'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse':12 , 'motorbike':13,
             'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


class VOCClassification(data.Dataset):
    """
    Args:
        root (string): Root directory of the VOC Dataset.
        config: data configurations
        split (string, optional): Select the image_set to use, ``train``, ``trainval`` ,
            ``val`` and "test"
        is_train(bool): train or evaluate
    """
    def __init__(self,
                 root,
                 cfg,
                 split='trainval',
                 is_train=True):
        self.num_class = 20
        self.is_train = is_train
        self.split = split
        self.type = cfg.MODEL.TYPE
        voc_root = os.path.join(root, cfg.DATADIR)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found')

        # load image and annotations list
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, split.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))
        # self.images = self.images[0:200]

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
        # label of multi-label classification
        anno = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        labels = []
        if 'name' in anno['annotation']['object']:
            labels.append(DATASET_CLASS_DICT[anno['annotation']['object']['name']])
        else:
            for object in anno['annotation']['object']:
                labels.append(DATASET_CLASS_DICT[object['name']])
                labels = list(np.sort(np.unique(labels)))
        target = torch.FloatTensor(self.num_class).zero_()
        target[labels] = 1

        # load image
        image = Image.open(self.images[index]).convert('RGB')
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
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

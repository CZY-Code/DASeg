import torch
import os 
import numpy as np
import cv2
import json
import random
from PIL import Image
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import yaml

from libs.dataset.transform import Normalize, ToTensor
from options import OPTION as opt
ROOT = "/home/chengzy/dataset/VIL-100-Seg"
DATA_CONTAINER = {}

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def convert_mask(mask, max_obj):
    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k) #数据变bool型
    oh = np.stack(oh, axis=2)
    return oh

class VDS(Dataset):
    def __init__(self, train=True, transform=None, samples_per_video=10):
        super().__init__()
        #每个视频在数据序列中被采样的次数
        self.sets = ['support', 'query']
        self.max_obj = 2
        self.transform = transform
        self.samples_per_video = samples_per_video if train else 1

        dbfile = os.path.join(ROOT, 'db_info.yaml')
        self.videos = os.listdir(os.path.join(ROOT, 'segs')) #长度为50的list
        self.videos.sort()
        # with open(dbfile, 'r') as f:
        #     db = yaml.load(f, Loader=yaml.Loader)['sequences']
        #     targetset = 'train' if train else 'test'
        #     self.videos = [info['name'] for info in db if info['set'] == targetset]
        if train:
            self.videos = self.videos[:40]
        else:
            self.videos = self.videos[40:]

        self.length = self.samples_per_video * len(self.videos)
        self.train = train
        self.normalize = Normalize()
        self.toTensor = ToTensor()

        if transform is not None:
            img_transforms = []
            for aug in transform:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def __getitem__(self, idx):
        vid = self.videos[(idx // self.samples_per_video)] #选择在哪个video里进行采样
        videofolder = os.path.join(ROOT, 'segs', vid)
        imgfolder = os.path.join(videofolder, 'JPEGImages')
        segfolder = os.path.join(videofolder, 'SegmentationClassPNG')
        imgs = os.listdir(imgfolder) #该视频下有标注的图像数量
        masks = os.listdir(segfolder)
        # index = random.sample(range(0,num_imgs), 2) #
        if self.train:
            support_idx, query_idx = np.random.randint(0, len(imgs), size=2) #可能重复的整数
        else:
            support_idx, query_idx = random.sample(range(0, len(imgs)), 2) #不重复的整数

        sample = dict()
        sample['vid'] = vid
        sample['support'] = dict()
        sample['support']['name'] = imgs[support_idx]
        sample['support']['img'] = np.array(Image.open(os.path.join(imgfolder, imgs[support_idx])))
        sample['support']['mask'] = np.array(Image.open(os.path.join(segfolder, masks[support_idx])))
        sample['support']['mask'] = convert_mask(sample['support']['mask'], self.max_obj)
        # set_mask = set(sample['support']['mask'].flatten().tolist()) #查看数组中所有不同的元素
        sample['query'] = dict()
        sample['query']['name'] = imgs[query_idx]
        sample['query']['img'] = np.array(Image.open(os.path.join(imgfolder, imgs[query_idx])))
        sample['query']['mask'] = np.array(Image.open(os.path.join(segfolder, masks[query_idx])))
        sample['query']['mask'] = convert_mask(sample['query']['mask'], self.max_obj)
        sample['size'] = sample['query']['img'].shape[:2]   

        for set in self.sets:
            mask_org = SegmentationMapsOnImage(sample[set]['mask'], shape=sample['size'])
            img_org = sample[set]['img']
            img, seg = self.transform(
                image=img_org.copy(), #.astype(np.uint8),
                segmentation_maps=mask_org)
            sample[set]['img'] = img
            sample[set]['mask'] = seg.get_arr().astype(np.float32) #此处的数据形式有待商榷

        self.normalize(sample)
        data = self.toTensor(sample) #此处字典换了
        # print(data['query']['mask'])
        return data
    
    def __len__(self):
        return self.length


def multibatch_collate_fn(batch):
    supp_imgs = torch.stack([sample['support']['img'] for sample in batch])
    qry_imgs = torch.stack([sample['query']['img'] for sample in batch])
    supp_masks = torch.stack([sample['support']['mask'] for sample in batch])
    qry_masks = torch.stack([sample['support']['mask'] for sample in batch])
    return supp_imgs, supp_masks, qry_imgs, qry_masks 
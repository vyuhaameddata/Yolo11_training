import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import random
from utils.datasets import InfiniteDataLoader, letterbox
from utils.torch_utils import torch_distributed_zero_first
import torch.nn.functional as F
vipsbin = r'D:\Shreyan\Development\yolo11\vips-dev-8.16\bin'
os.add_dll_directory(vipsbin)
# add_dll_dir = getattr(os, 'add_dll_directory', None)
# if callable(add_dll_dir):
#     add_dll_dir(vipsbin)
# else:
#     os.environ['PATH'] = os.pathsep.join((vipsbin, os.environ['PATH']))

import ctypes
ctypes.WinDLL(r"D:\Shreyan\Development\yolo11\vips-dev-8.16\bin\libvips-42.dll")
import pyvips


def create_dataloader_custom(path, imgsz, stride, opt, batch_size, tile_size, overlap,
                              hyp=None, augment=False, rect=False, pad=0.0, rank=-1, world_size=1,
                              workers=4, image_weights=False, quad=False):
    print('batch_size', batch_size)
    with torch_distributed_zero_first(rank):
        dataset = load_custom(path, imgsz, batch_size=batch_size, tile_size=tile_size, overlap=overlap,
                              rect=rect, single_cls=opt.single_cls, stride=int(stride), pad=pad)
    
    batch_size = opt.batch_size
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    
    loader = DataLoader if image_weights else InfiniteDataLoader
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=load_custom.collate_fn4 if quad else load_custom.collate_fn)
    
    return dataloader, dataset

class load_custom(Dataset):
    def __init__(self, path, img_size, tile_size, overlap, batch_size=16, single_cls=False, rect=False,
                 stride=32, pad=0.0):
        self.img_size = img_size
        self.rect = rect
        self.stride = stride
        self.path = path
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.img_files = []
        
        slide = pyvips.Image.new_from_file(path, access="sequential")
        self.wsi_width = slide.width
        self.wsi_height = slide.height
        
        cols = (self.wsi_width - overlap) // (tile_size - overlap)
        rows = (self.wsi_height - overlap) // (tile_size - overlap)
        
        for row in range(rows):
            for col in range(cols):
                self.img_files.append((col, row))
        
        self.n = len(self.img_files)
        self.indices = range(self.n)
        print(f'Tiles: {self.n}, Batch size: {batch_size}, Number of batches: {self.n // batch_size}')
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        index = self.indices[index]
        img, (h0, w0), (h, w), coord = load_tile(self, index)
        img, ratio, pad = letterbox(img, self.img_size, auto=False)
        shapes = (h0, w0)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), self.img_files[index], shapes, coord
    
    def collate_fn(batch):
        img, indices, shapes, coords = zip(*batch)
        return torch.stack(img, 0), shapes, coords
    
    def collate_fn4(batch):
        img, indices, shapes, coords = zip(*batch)
        n = len(shapes) // 4
        img4, shapes4, coords4 = [], shapes[:n], coords[:n]
        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[0].type(img[i].type())
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
            img4.append(im)
        return torch.stack(img4, 0), shapes4, coords4

def load_tile(self, index):
    tile_id = self.img_files[index]
    col, row = tile_id
    x = col * (self.tile_size - self.overlap)
    y = row * (self.tile_size - self.overlap)
    
    slide = pyvips.Image.new_from_file(self.path, access="sequential")
    img = slide.crop(x, y, self.tile_size, self.tile_size).numpy()
    coord = (x, y)
    h0, w0 = img.shape[:2]
    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
    
    return img, (h0, w0), img.shape[:2], coord

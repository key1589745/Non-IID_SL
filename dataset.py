# -*- coding:utf-8 -*-
import random, glob
import numpy as np
import nibabel as nib
from batchgenerators.dataloading.dataset import Dataset
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading import MultiThreadedAugmenter
from augmentations import get_DA

def train_val_split(dataroot):
    
    imgs = np.load(dataroot+'/imgs.npy', allow_pickle=True)
    labs = np.load(dataroot+'/labs.npy', allow_pickle=True)
    
    rand_seed = random.randint(0,100)
    random.seed(rand_seed)
    random.shuffle(imgs)
    random.seed(rand_seed)
    random.shuffle(labs)
   
    test_split = len(imgs) // 3
    train_split = len(imgs) // 2

    return ({'imgs':imgs[:train_split],'labs':labs[:train_split]}, \
            {'imgs':imgs[train_split:-test_split],'labs':labs[train_split:-test_split]}, \
            {'imgs':imgs[-test_split:],'labs':labs[-test_split:]})



class CMRDataLoader(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, label=None, seed_for_shuffle=1,
                 return_incomplete=False, shuffle=True, infinite=False):

        super(CMRDataLoader, self).__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        
        self.imgs = np.concatenate(data['imgs'],axis=0)
        self.labs = np.concatenate(data['labs'],axis=0)
        
        if label is not None:
            if 0 not in label:
                temp = np.zeros_like(self.labs)
                for i in label:
                    temp += (self.labs==i) * i
                self.labs = temp
            else:
                self.labs = (self.labs!=0)
                
        self.indices = np.arange(len(self.imgs))
        self.patch_size = self.imgs.shape[-2:]

    def generate_train_batch(self):
        indices = self.get_indices()
        data = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        for i, idx in enumerate(indices):
            data[i][0] = self.imgs[idx]
            seg[i][0]  = self.labs[idx]
            
        return {'data': data, 'seg':seg}


def getTrainLoader(data,configs,label):
    dataloader_train = CMRDataLoader(data, configs.batch_size, configs.num_workers, label=label, infinite=True)
    transforms = get_DA((configs.size,configs.size), spatial_DA=True, intensity_DA=False)
    tr_gen = MultiThreadedAugmenter(dataloader_train, transforms, num_processes=configs.num_workers,
                                    num_cached_per_queue=3,
                                    seeds=None, pin_memory=False)
    return tr_gen

def getValiLoader(data, configs):

    return CMRDataLoader(data, 32, configs.num_workers, shuffle=False)

def getTestLoader(data):
    
    test_loader = []
    
    for imgs, seg in zip(data['imgs'],data['labs']):
        test_loader.append({'data': np.expand_dims(imgs,1).astype(np.float32), 'seg':seg.astype(np.float32)})

    return test_loader
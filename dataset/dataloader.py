import torch
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from .augmentation import *
from sklearn.model_selection import train_test_split

class customDataset(Dataset):
    def __init__(self, args, df, label, transform):
        self.args = args
        self.df = df
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, :]["path"]
        trued = os.path.join(self.args.img_path ,img_path)
        cv2_img = cv2.imread(trued, cv2.IMREAD_COLOR)
        
        img = self.transform(cv2_img)
        if self.label is not None:
            label = self.label.iloc[idx, :]
            label = np.array(label)
            return img, label # , idx
        else:
            return img

    def __len__(self,):
        return len(self.df)

def get_kfoldloader(args, train_X, train_Y, valid_X, valid_Y):
    train_aug, valid_aug = get_aug(args)
    train_dataset = customDataset(args, train_X, train_Y, train_aug)   
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)                                 
    train_dl = DataLoader(train_dataset, batch_size = args.batchsize, num_workers = 4 , shuffle = False, sampler = train_sampler, pin_memory = True)

    valid_dataset = customDataset(args, valid_X, valid_Y, valid_aug)     # valid Dataset
    valid_dl = DataLoader(valid_dataset, batch_size = args.batchsize, num_workers = 4, shuffle = False, pin_memory = True)

    return train_dl, valid_dl, train_sampler

def get_dataloader(args):
    train_df = pd.read_csv("/home/psboys/psboys/data_iccvw/train.csv")
    train_aug, valid_aug = get_aug(args)
    train_X, train_Y = train_df.iloc[ : , :6], train_df.iloc[ : , 6:]
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size = args.split, random_state = args.seed)

    train_dataset = customDataset(args, train_X, train_Y, train_aug)   
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)                                 
    train_dl = DataLoader(train_dataset, batch_size = args.batchsize, num_workers = 4 , shuffle = False, sampler = train_sampler, pin_memory = True)

    valid_dataset = customDataset(args, valid_X, valid_Y, valid_aug)     # valid Dataset
    valid_dl = DataLoader(valid_dataset, batch_size = args.batchsize, num_workers = 4, shuffle = False, pin_memory = True)
    
    return train_dl, valid_dl, train_sampler

def get_testloader(args):
    dev_df = pd.read_csv("/home/psboys/psboys/data_iccvw/test.csv")
    submit_df = pd.read_csv("/home/psboys/psboys/data_iccvw/sample_submission.csv")
    test_df = pd.merge(dev_df, submit_df)
    _, valid_aug = get_aug(args)

    test_ds = customDataset(args = args, df = test_df, label = None, transform = valid_aug)
    test_dl = DataLoader(test_ds, batch_size = args.batchsize, num_workers = 4, shuffle = False)

    return test_dl, submit_df
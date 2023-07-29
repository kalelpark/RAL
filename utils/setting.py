import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelPrecision
from libauc.metrics import auc_roc_score

def init_cuda_distributed(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    torch.distributed.init_process_group( backend='nccl', init_method='env://')
    args.local_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size() 

    args.is_master = args.local_rank == 0    
    args.device = torch.device(f'cuda:{args.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(args.local_rank)
    seed_everything(args.seed + args.local_rank)

# Set Seed
def seed_everything(seed):              
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

def metrics(pred, label):
    metric_precision = MultilabelPrecision(num_labels = 26, average = "macro")
    precision = metric_precision(pred, label)
    auc_score = np.mean(auc_roc_score(label, pred))

    pred = pred > 0.5
    acc = (pred == label).float().mean()
    
    return precision, acc, auc_score

def save_model(args, model, idx = None):
    torch.save(model.state_dict(), os.path.join(args.model_path, args.model + f"_{args.img_size}" + args.store_name + ".pt"))
    torch.save(model.module.state_dict(), os.path.join(args.model_path, args.model + f"_{args.img_size}_module(X)_" + args.store_name + ".pt"))
    
    

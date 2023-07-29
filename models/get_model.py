import timm
import torch
from torch.nn.parallel import DistributedDataParallel
import timm
import torch

from .part import *
from .nan import *
from .convout import *

def get_model(args):
    if args.model == "part":
        model = part_model(args, 26)
    elif args.model == "convout":
        model = convout()
    elif args.model == "resnet50":
        model = timm.create_model("resnet50", pretrained = True, num_classes = 26)
    elif args.model == "convnext":
        model = timm.create_model('convnext_base', pretrained=True, num_classes = 26)
        
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    model = DistributedDataParallel(model, static_graph=False, device_ids=[args.local_rank], find_unused_parameters = False)
    return model
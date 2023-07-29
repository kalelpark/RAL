import torch
import warnings
warnings.filterwarnings("ignore")

from utils import *
from models import *
from dataset import *
from train import *
from predict import *

if __name__ == "__main__":
    args = get_args_parser()
    init_cuda_distributed(args)
    net = get_model(args)
    
    if args.train:
        optimizer, criterion = make_loss_optimizer(args, net)
        train_dl, valid_dl, train_sampler = get_dataloader(args)
        train(args, net, optimizer, criterion, train_dl, valid_dl, train_sampler, 1)

    # if args.infer:
    #     test_dl, submit_df = get_testloader(args)
    #     infer(args, net, test_dl, submit_df)
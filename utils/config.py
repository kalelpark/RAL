import argparse
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser("MIMIC2.0, CXR-LT: Multi-Label Long-Tailed Classification", add_help=False)
    parser.add_argument("--seed", type = int, required = False, default = 0)
    parser.add_argument("--gpu_ids", type = str)
    parser.add_argument("--img_size", type = int, required = True, default = 448)       # Fill in Size

    # train or infer
    parser.add_argument("--train", type = int, required = False, default = 0)
    parser.add_argument("--infer", type = int, required = False, default = 0)
    parser.add_argument("--split", type = float, required = False, default = 0.05)
    
    # model
    parser.add_argument("--model", type = str, required = False)
    parser.add_argument("--OPTIMIZER", type = str, default = "Adam", required = False)
    parser.add_argument("--criterion", type = str, default = "Ral", required = False)
    parser.add_argument("--batchsize", type = int, default= 256, required = False)
    parser.add_argument("--epochs", type = int, default= 60, required = False)
    parser.add_argument("--lr", type = float, default= 1e-4, required = False)
    parser.add_argument("--BIAS_LR_FACTOR", type = float, default= 1.0, required = False)
    parser.add_argument("--WEIGHT_DECAY", type = float, default = 0.0005, required = False)
    parser.add_argument("--WEIGHT_DECAY_BIAS", type = float, default = 0.0005, required = False)
    
    # PART Model Setting
    parser.add_argument("--LAST_STRIDE", type = int, default = 1, required = False)
    parser.add_argument("--PRETRAIN_PATH", type = str, default = "wongi", required = False)
    parser.add_argument("--COS_LAYER", type = bool, default = False, required = False)
    parser.add_argument("--RERANKING", type = bool, default = True, required = False)
    parser.add_argument("--MODEL_NAME", type = str, default = "resnet50", required = False)
    parser.add_argument("--PRETRAIN_CHOICE", type = str, default = "imagenet", required = False)
    parser.add_argument("--NUM_SELECT_PART", type = int, default = 4, required = False)
    parser.add_argument("--NUM_PART_STACK", type = int, default = 64, required = False)
    parser.add_argument("--HARD_FACTOR", type = float, default = 0.2, required = False)
    parser.add_argument("--TRAIN_BN_MOM", type = float, default = 0.1, required = False)

    parser.add_argument("--STEPS", action='append', default = [8, 16, 24,], required=False)
    parser.add_argument("--GAMMA", type = float, default = 0.1, required = False)
    parser.add_argument("--WARMUP_FACTOR", type = float, default = 0.01, required = False)
    parser.add_argument("--WARMUP_EPOCHS", type = float, default = 4, required = False)
    parser.add_argument("--WARMUP_METHOD", type = str, default = "linear", required = False)
    
    # path
    parser.add_argument("--img_path", type = str, default = "/home/psboys/psboys/224", required = False)
    parser.add_argument("--model_path", type = str, default = "chkpt", required = False)
    # parser.add_argument("--store_name", type = str, help = "identify Name", default = "xXx", required = True)
    parser.add_argument("--csv_path", type = str, default = "largemit", required = False)
    parser.add_argument("--save_model", type = int, default = 0, required = False)
    
    return parser.parse_args()


# 
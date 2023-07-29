import torch
import time
import wandb
from utils import *

def train(args, model, optimizer, loss_fn, train_dl, valid_dl, train_sampler, num): 
    best_precision = -1
    if args.is_master:
        print()
        wandb.init(name = f"{args.model}", project = "CXR Train & Split ConvNext Series", reinit = True, entity = "psboys", config = args)
        print("Stat Train and Valid")

    scaler = torch.cuda.amp.GradScaler(enabled = True)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if args.is_master:
            print(f"Train : [ {epoch + 1} / Epoch ]")
    
        start = time.time()
        model.train()
        for img, label in train_dl:
            optimizer.zero_grad()
            img, label = img.to(args.device), label.to(args.device)
            with torch.cuda.amp.autocast(enabled = True):    
                pred = model(img)                
                loss = loss_fn(pred, label.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # print("hello")
            torch.cuda.synchronize()
        # scheduler.step()
        
        pred_list, label_list = torch.Tensor([]), torch.Tensor([])
        test_loss = 0

        if args.is_master:
            print(f"Eval : [ {epoch + 1} / Epoch ]")
        model.eval()
        for img, label in valid_dl:         # # valid code
            with torch.no_grad():
                img, label = img.to(args.device), label.to(args.device)
                pred_v2 = model(img)
                loss = loss_fn(pred_v2, label.float())
                pred = torch.sigmoid(pred_v2)
            test_loss += loss
            pred_list = torch.cat([pred_list.cpu(), pred.cpu()], dim = 0)
            label_list = torch.cat([label_list.cpu(), label.cpu()], dim = 0)

        end = time.time()               
        precision, acc, auc_score = metrics(pred_list, label_list)

        if args.save_model and best_precision < precision and args.is_master:
            torch.cuda.synchronize()
            best_precision = precision
            save_model(args, model, idx = num)
            print("save")

        if args.is_master:
            print(f"{end - start:.5f} sec")
            print(f"Accuracy : {acc} precision : {precision} Loss : {test_loss}, AUC Score : {auc_score}")
            wandb.log({
                "Valid_Acc" : acc,
                "Valid_Precision" : precision,
                "Loss" : test_loss, 
                "Valid_Auc_score" : auc_score
            })
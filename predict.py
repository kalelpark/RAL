import torch
import os

def infer(args, model, test_dl, submit_df):
    
    if args.is_master:
        print("model Evaluate")
    
    for idx in range(10):
        if args.is_master:
            print(f"start {idx}")
        
        model.load_state_dict(torch.load(f"fchkpt/convnext_1024_{idx}_foldingclear.pt", map_location = args.device))
        model.eval() 
            
        batch_index = 0
        for i, images in enumerate(test_dl):
            images = images.to(args.device)
            with torch.no_grad():
                pred = model(images)
                outputs = torch.sigmoid(pred)

            batch_index = i * args.batchsize
            submit_df.iloc[batch_index:batch_index+args.batchsize, 1:] = \
            outputs.float().squeeze(0).detach().cpu().numpy()
        if args.is_master:
            submit_df.to_csv(os.path.join(args.csv_path, args.model + f"_{idx}" + ".csv"), index=False)
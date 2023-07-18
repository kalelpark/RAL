# # Train
# torchrun --nproc_per_node=8 main.py --gpu_ids 0,1,2,3,4,5,6,7 --seed 0 --train 1 --model convnext --batchsize 64 --epochs 30

# # Final Train
# torchrun --nproc_per_node=8 main.py --gpu_ids 0,1,2,3,4,5,6,7 --seed 0 --img_size 384 --infer 1 --model convnext --batchsize 17 --epochs 200 --save_model 1

torchrun --nproc_per_node=8 main.py --gpu_ids 0,1,2,3,4,5,6,7 --seed 0 --img_size 384 --infer 1 --model convnext --batchsize 15 --epochs 200 --save_model 1


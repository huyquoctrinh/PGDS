# Swin Base
CUDA_VISIBLE_DEVICES=0 python train.py --config_file /storage/huytq14/multpersonTracking/SOLIDER-REID/configs/market/swin_tiny_normal.yaml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH '/storage/huytq14/multpersonTracking/SOLIDER-REID/base_weights/swin_tiny_msmt17.pth' OUTPUT_DIR './log/full_data/swin_tiny_base' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2

# Swin Small
#CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msmt17/swin_small.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'path/to/SOLIDER/log/lup/swin_small/checkpoint_tea.pth' OUTPUT_DIR './log/msmt17/swin_small' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2

# Swin Tiny
#CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msmt17/swin_tiny.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'path/to/SOLIDER/log/lup/swin_tiny/checkpoint_tea.pth' OUTPUT_DIR './log/msmt17/swin_tiny' SOLVER.BASE_LR 0.0008 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2

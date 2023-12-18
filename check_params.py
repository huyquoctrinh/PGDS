import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from thop import profile, clever_format
import torch 
from fvcore.nn import FlopCountAnalysis
import numpy as np 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/home/nhdang/ReID_backup/configs/market/swin_tiny.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)

        print(model)

    input_data = torch.randn(1, 3, 384, 128)
    input_data = input_data.to("cuda:0")

# Load a pre-trained model (for example, ResNet18)
# model = models.resnet18()

# Measure FLOPs
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    model.eval()
    flops_old, params = profile(model.to("cuda:0"), inputs=(input_data,))
    # model.eval()
    # Format the results for better readability
    flops_old, params = clever_format([flops_old, params], "%.3f")

    flops = FlopCountAnalysis(model.to("cuda:0"), (input_data,))

    print("Flop fvcore:",flops.total())
    print(f"MAdds: {round(flops.total() * 1e-9, 2)} G")

    print(f"FLOPs: {flops_old}")
    print(f"Params: {params}")
    print(f"Trainable param: {trainable_params}")

    # if cfg.DATASETS.NAMES == 'VehicleID':
    #     for trial in range(10):
    #         train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    #         rank_1, rank5 = do_inference(cfg,
    #              model,
    #              val_loader,
    #              num_query)
    #         if trial == 0:
    #             all_rank_1 = rank_1
    #             all_rank_5 = rank5
    #         else:
    #             all_rank_1 = all_rank_1 + rank_1
    #             all_rank_5 = all_rank_5 + rank5

    #         logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
    #     logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    # else:
    #    do_inference(cfg,
    #              model,
    #              val_loader,
    #              num_query)


import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from .fusion import Fusion
# from kmeans_pytorch import kmeans
from pytorch_metric_learning import losses

def do_train(cfg,
             model,
             pose_model,
             center_criterion,
             koleo_criterion,
             arc_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    cls_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction = 'batchmean')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        pose_model.to(local_rank)
        arc_criterion.to(local_rank)
        # cls_loss.to(local_rank)
        # fusion = Fusion(768).to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            pose_model = torch.nn.parallel.DistributedDataParallel(pose_model, device_ids=[local_rank], find_unused_parameters=True)
            

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    scaler1 = amp.GradScaler()
    # train
    pose_opt = torch.optim.SGD(pose_model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        pose_model.eval()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            pose_opt.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            
            with amp.autocast(enabled=True):
                
                score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view ) 
                
                pose_score, pose_feat = pose_model(img)
                
                
                divergence_loss = kl_loss(torch.nn.functional.softmax(pose_feat,1)/2, torch.nn.functional.softmax(feat,1)/2)
                # pose_loss = 0.8*loss_fn(pose_score, pose_feat, target, target_cam) + 0.2 * arc_criterion(pose_score, target)
                
                loss = 0.8*loss_fn(score, feat, target, target_cam) + 0.2*divergence_loss
                # print(loss)
                
            # pose_loss.backward(retain_graph=True)
            # pose_opt.step()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            # if isinstance(pose_score, list):
            #     print(pose_score.max(1)[1], target)
            #     pose_acc = (pose_score[0].max(1)[1] == target).float().mean()
            # else:
            #     # print(pose_score)
            #     print(torch.max(pose_score, 1), target)
            #     pose_acc = (torch.max(pose_score, 1) == target).sum()/64

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Loss_pose: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f},  Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg,base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _ = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat , _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



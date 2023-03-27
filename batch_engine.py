import math
import time
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.distributed import reduce_tensor
from tools.utils import AverageMeter, to_scalar, time_str
from tools.utils import data_prefetcher


def batch_trainer(cfg, args, epoch, model, train_loader, criterion, optimizer, scheduler=None):
    
    model.train()
    epoch_time = time.time()

    train_loader = data_prefetcher(train_loader)
    loss_meter_cls = AverageMeter()
    if cfg.NAME == 'OOD':
        loss_meter_rec_0 = AverageMeter()
        loss_meter_rec_1 = AverageMeter()
        
    batch_num = len(train_loader)
    gt_list = np.zeros([batch_num, cfg.TRAIN.BATCH_SIZE])
    preds_logits = np.zeros([batch_num, cfg.TRAIN.BATCH_SIZE, args.cls_num])
 
    lr = optimizer.param_groups[0]['lr']

    imgs, gt_label = train_loader.next()
    step = 0
    while imgs is not None:
        batch_time = time.time()
        
        if cfg.NAME == 'ID':
            train_logits = model(imgs)
            train_loss = criterion(train_logits, gt_label, None, None, None, None)
        else:
            train_logits, x_0, x_1, rec_0, rec_1 = model(imgs)
            cls_loss, rec_0_loss, rec_1_loss = criterion(train_logits, gt_label, x_0, x_1, rec_0, rec_1)
            train_loss = cls_loss + rec_0_loss + rec_1_loss

        optimizer.zero_grad()
        train_loss.backward()

        if cfg.TRAIN.CLIP_GRAD:
            clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        
        if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' or cfg.TRAIN.LR_SCHEDULER.TYPE == 'cosine_annealing':
            scheduler.step()

        torch.cuda.synchronize()

        if cfg.NAME == 'ID':
            loss_meter_cls.update(to_scalar(reduce_tensor(train_loss, args.world_size) if args.distributed else train_loss))
        else:
            loss_meter_cls.update(to_scalar(reduce_tensor(cls_loss, args.world_size) if args.distributed else cls_loss))
            loss_meter_rec_0.update(to_scalar(reduce_tensor(rec_0_loss, args.world_size) if args.distributed else rec_0_loss))
            loss_meter_rec_1.update(to_scalar(reduce_tensor(rec_1_loss, args.world_size) if args.distributed else rec_1_loss))
        
        gt_list[step] = gt_label.cpu().numpy()
        preds_logits[step] = train_logits.detach().cpu().numpy()

        log_interval = 100

        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            if args.local_rank == 0:
                print(f'{time_str()}, '
                      f'Step {step}/{batch_num} in Ep {epoch}, '
                      f'LR: [{optimizer.param_groups[0]["lr"]:.1e}] '
                      f'Time: {time.time() - batch_time:.2f}s , '
                      f'train_loss: {loss_meter_cls.avg:.4f}, ')
                
        step += 1
        imgs, gt_label = train_loader.next()

    if cfg.NAME == 'ID':
        train_loss = [loss_meter_cls.avg]
        if args.local_rank == 0:
            print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter_cls.avg:.4f}')
    else:
        train_loss = [loss_meter_cls.avg, loss_meter_rec_0.avg, loss_meter_rec_1.avg]
        if args.local_rank == 0:
            print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Cls_Loss: {loss_meter_cls.avg:.4f}, rec_Loss: {loss_meter_rec_0.avg:.4f}, {loss_meter_rec_1.avg:.4f}')
    
    gt_list = np.reshape(gt_list, [batch_num * cfg.TRAIN.BATCH_SIZE, 1])
    preds_logits = np.reshape(preds_logits, [batch_num * cfg.TRAIN.BATCH_SIZE, args.cls_num])
    
    return train_loss, gt_list, preds_logits


def valid_trainer(cfg, args, epoch, model, valid_loader_list, criterion):
    
    model.eval()
    
    if cfg.NAME == 'ID':
        loss_meter = AverageMeter()
        
        preds_logits = []
        gt_list = []
        
        valid_loader = valid_loader_list[0]
        with torch.no_grad():
            for step, (imgs, gt_label) in enumerate(valid_loader):
                imgs = imgs.cuda(non_blocking=cfg.TRAIN.NON_BLOCKING)
                
                gt_label = gt_label.cuda(non_blocking=cfg.TRAIN.NON_BLOCKING)
                valid_logits = model(imgs)

                valid_loss = criterion(valid_logits, gt_label, None, None, None, None)

                gt_list.append(gt_label.cpu().numpy())
                preds_logits.append(valid_logits.cpu().numpy())
                loss_meter.update(to_scalar(reduce_tensor(valid_loss, args.world_size) if args.distributed else valid_loss))

                torch.cuda.synchronize()

        valid_loss = loss_meter.avg
        gt_label = np.concatenate(gt_list, axis=0)
        preds_logits = np.concatenate(preds_logits, axis=0)

        return valid_loss, gt_label, preds_logits
        
    else:
        if cfg.DATASET.TYPE == 'cifar':
            loss_meter_cls = AverageMeter()
            loss_meter_rec_0 = AverageMeter()
            loss_meter_rec_1 = AverageMeter()

            preds_logits = []
            gt_list = []
        
            #ID,isun,lsuncrop,lsunre,tinycrop,tinyre
            preds_all = [[],[],[],[],[],[]]
            
            count = 0
            with torch.no_grad():
                for valid_loader in valid_loader_list:
                    
                    probs_list = []
                    rec_0_list = []
                    rec_1_list = []
                    
                    for step, (imgs, gt_label) in enumerate(valid_loader):
                        
                        imgs = imgs.cuda(non_blocking=cfg.TRAIN.NON_BLOCKING)
                        gt_label = gt_label.cuda(non_blocking=cfg.TRAIN.NON_BLOCKING)
                        valid_logits, x_0, x_1, rec_0, rec_1 = model(imgs)

                        if count == 0:
                            cls_loss, rec_0_loss, rec_1_loss = criterion(valid_logits, gt_label, x_0, x_1, rec_0, rec_1)
                            
                            gt_list.append(gt_label.cpu().numpy())
                            preds_logits.append(valid_logits.cpu().numpy())
                            loss_meter_cls.update(to_scalar(reduce_tensor(cls_loss, args.world_size) if args.distributed else cls_loss))
                            loss_meter_rec_0.update(to_scalar(reduce_tensor(rec_0_loss, args.world_size) if args.distributed else rec_0_loss))
                            loss_meter_rec_1.update(to_scalar(reduce_tensor(rec_1_loss, args.world_size) if args.distributed else rec_1_loss))

                        probs = F.softmax(valid_logits/1000, dim=-1)
                        
                        norm_0 = torch.norm(x_0, p=2, dim=-1, keepdim=True)
                        rec_0 = torch.sum(((x_0/norm_0) - (rec_0/norm_0))**2,dim=-1)
                        
                        norm_1 = torch.norm(x_1, p=2, dim=-1, keepdim=True)
                        rec_1 = torch.sum(((x_1/norm_1) - (rec_1/norm_1))**2,dim=-1)
                        
                        probs_list.append(probs.cpu().numpy())
                        rec_0_list.append(rec_0.cpu().numpy())
                        rec_1_list.append(rec_1.cpu().numpy())
                        
                        torch.cuda.synchronize()
                    
                    if count == 0:
                        valid_loss = [loss_meter_cls.avg, loss_meter_rec_0.avg, loss_meter_rec_1.avg]
                        gt_list = np.concatenate(gt_list, axis=0)
                        preds_logits = np.concatenate(preds_logits, axis=0)
                    
                    probs_list = np.concatenate(probs_list, axis=0)
                    rec_0_list = np.concatenate(rec_0_list, axis=0)
                    rec_1_list = np.concatenate(rec_1_list, axis=0)
                    
                    preds_all[count].append(probs_list) 
                    preds_all[count].append(rec_0_list) 
                    preds_all[count].append(rec_1_list) 
                        
                    count+=1
                    
            return valid_loss, gt_list, [preds_logits, preds_all]
            


import time

import numpy as np
from easydict import EasyDict
import torch
import scipy.stats as stats 
from scipy.stats import norm
import libmr
import seaborn as sns


def get_metrics_values_list(labels, predictions, num_thresholds=100):
    
    start = 0
    step = 1.0 / num_thresholds
    end = 1 + step
    pre_ls = []
    rec_ls = []
    tpr_ls = []
    fpr_ls = []
    
    while start <= end:
        pre, rec, tpr, fpr,_ = calculate_metrics(labels=labels, predictions=predictions, threshold=start)
        pre_ls.append(pre)
        rec_ls.append(rec)
        tpr_ls.append(tpr)
        fpr_ls.append(fpr)
        start = start + step

    fpr_tpr_coordinate = []
    for fpr, tpr in sorted(zip(fpr_ls, tpr_ls), reverse=True):
        fpr_tpr_coordinate.append((fpr, tpr))

    fpr_tpr_coordinate_unique = list(set(fpr_tpr_coordinate))
    fpr_tpr_coordinate_unique.sort(key=fpr_tpr_coordinate.index)
    fpr_ls = [point[0] for point in fpr_tpr_coordinate_unique]
    tpr_ls = [point[1] for point in fpr_tpr_coordinate_unique]

    rec_pre_coordinate = []
    for rec, pre in sorted(zip(rec_ls, pre_ls), reverse=True):
        rec_pre_coordinate.append((rec, pre))

    rec_pre_coordinate_unique = list(set(rec_pre_coordinate))
    rec_pre_coordinate_unique.sort(key=rec_pre_coordinate.index)
    rec_ls = [point[0] for point in rec_pre_coordinate_unique]
    pre_ls = [point[1] for point in rec_pre_coordinate_unique]

    return rec_ls, pre_ls, fpr_ls, tpr_ls

def get_auroc(fpr_ls, tpr_ls):
    auc_value = 0.0
    for ix in range(len(fpr_ls[:-1])):
        x_right, x_left = fpr_ls[ix], fpr_ls[ix+1]
        y_top, y_bottom = tpr_ls[ix], tpr_ls[ix+1]
        temp_area = abs(x_right-x_left) * (y_top + y_bottom) * 0.5
        auc_value += temp_area
    return auc_value


def get_aupr(rec_ls, pre_ls):
    pr_value = 0.0
    for ix in range(len(rec_ls[:-1])):
        x_right, x_left = rec_ls[ix], rec_ls[ix + 1]
        y_top, y_bottom = pre_ls[ix], pre_ls[ix + 1]
        temp_area = abs(x_right-x_left) * (y_top + y_bottom) * 0.5
        pr_value += temp_area
    return pr_value


def area_curves(labels, predictions):
    rec_ls, pre_ls, fpr_ls, tpr_ls = get_metrics_values_list(labels=labels, predictions=predictions)
    auc_roc = np.around(get_auroc(fpr_ls, tpr_ls), 4)
    auc_pr = np.around(get_aupr(rec_ls, pre_ls), 4)
    return auc_roc, auc_pr


def calculate_metrics(labels, predictions, threshold=0.5):

    pos_label_index = np.where(labels == 1)[0]                  
    pos_label_count = len(np.where(labels == 1)[0])                 

    predict_label_index = np.where(predictions >= threshold)[0]     
    index_scores = []                                               
    for pred_score, index in sorted(zip(predictions[predict_label_index], predict_label_index), reverse=True):
        index_scores.append((index, pred_score))

    top_k = len(predict_label_index)
    top_index_score = index_scores[:top_k]
    correct_predict = 0
    for item in top_index_score:
        if item[0] in pos_label_index:
            correct_predict += 1

    try:
        TP = correct_predict
        FN = pos_label_count - TP
        FP = len(top_index_score) - TP
        TN = len(labels) - pos_label_count - FP

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        tpr = recall
        fpr = FP / (TN + FP)
        tnr = TN / (TN + FP)

    except ZeroDivisionError:
        
        precision, recall, tpr, fpr,tnr = 0, 0, 0, 0, 0

    return precision, recall, tpr, fpr,tnr


def metrics(preds_probs, gt_label, cfg=None):
    
    result = EasyDict()

    for thres in np.linspace(0,1,100):
        precision, recall, tpr, fpr, tnr = calculate_metrics(labels=gt_label, predictions=preds_probs, threshold=thres)
        if tpr <= 0.95:
            result.fpr = np.around(fpr, decimals=4)
            break

    detection_error = 1        
    for thres in np.linspace(0,1,100):
        precision, recall, tpr, fpr, tnr = calculate_metrics(labels=gt_label, predictions=preds_probs, threshold=thres)
        if 0.5*(1 - tpr) + 0.5*fpr < detection_error:
            detection_error = 0.5*(1 - tpr) + 0.5*fpr
    result.de = np.around(detection_error, decimals=4)
    
    auc_roc, auc_pr = area_curves(labels=gt_label, predictions=preds_probs)
    result.roc = auc_roc
    result.pr = auc_pr
    
    return result

def get_ood_metrics(preds_all, cfg=None):
    
    prob_ID = np.max(preds_all[0][0],-1)
    rec_0_ID = preds_all[0][1]
    rec_1_ID = preds_all[0][2]
    
    loc_p,scale_p = norm.fit(prob_ID)
    prob_ID = norm.cdf(prob_ID,loc=loc_p,scale=scale_p*cfg.TRAIN.EPSILON)
    
    loc_0,scale_0 = norm.fit(rec_0_ID)
    rec_0_ID = 1 - norm.cdf(rec_0_ID,loc=loc_0,scale=scale_0*cfg.TRAIN.EPSILON)
    
    loc_1,scale_1 = norm.fit(rec_1_ID)
    rec_1_ID = 1 - norm.cdf(rec_1_ID,loc=loc_1,scale=scale_1*cfg.TRAIN.EPSILON)
    
    score_ID = prob_ID * rec_0_ID * rec_1_ID

    result = []
    
    for set_index in range(1,len(preds_all)):

        prob_OOD = np.max(preds_all[set_index][0],-1)
        rec_0_OOD = preds_all[set_index][1]
        rec_1_OOD = preds_all[set_index][2]
        
        prob_OOD = norm.cdf(prob_OOD,loc=loc_p,scale=scale_p*10)
        rec_0_OOD = 1 - norm.cdf(rec_0_OOD,loc=loc_0,scale=scale_0*10)
        rec_1_OOD = 1 - norm.cdf(rec_1_OOD,loc=loc_1,scale=scale_1*10)
        
        score_OOD = prob_OOD * rec_0_OOD * rec_1_OOD
        
        probs = np.concatenate([score_ID, score_OOD],0)
        probs = probs/np.max(probs)
        gt_label = np.concatenate([np.ones(score_ID.shape),np.zeros(score_OOD.shape)],0)
    
        result.append(metrics(probs, gt_label, cfg=None))
        
    return result
    
    
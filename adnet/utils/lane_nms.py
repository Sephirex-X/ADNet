from adnet.ops import nms
import math
import cv2
import torch
import numpy as np
import torch.nn as nn


def nms_enter(batch_proposals, nms_thres, nms_topk, conf_threshold,device = 'cpu'):
    softmax = nn.Softmax(dim=1)
    proposals_list = []
    for proposals in batch_proposals:
        # anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
        # The gradients do not have to (and can't) be calculated for the NMS procedure
        with torch.no_grad():
            scores = softmax(proposals[:, :2])[:, 1]
            if conf_threshold is not None:
                # apply confidence threshold
                above_threshold = scores > conf_threshold
                # Note: to avoid Nonzero op
                # true_proposals = []
                # for index in range(above_threshold.shape[0]):
                #     if above_threshold[index]:
                #         true_proposals.append(proposals[index])
                # proposals = torch.stack(true_proposals)
                # end
                proposals = proposals[above_threshold]
                scores = scores[above_threshold]
                # anchor_inds = anchor_inds[above_threshold]
            if proposals.shape[0] == 0:
                proposals_list.append((proposals[[]]))
                continue
            if 'cuda' in device:
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
            # to generate onnx model use the code below, however its way more slower!
            elif 'cpu' in device:
                keep, num_to_keep = Lane_nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
            keep = keep[:num_to_keep]
        proposals = proposals[keep]
        # anchor_inds = anchor_inds[keep]
        # attention_matrix = attention_matrix[anchor_inds]
        proposals_list.append((proposals, 0, 0, 0))
        # proposals_list.append((proposals))
    return proposals_list
    # start <rewrite nms for converting onnx model> NOTE! its way more slower


def Lane_nms(proposals, scores, overlap=50, top_k=4):
    keep_index = []
    sorted_score, indices = torch.sort(scores, descending=True)  # from big to small
    r_filters = np.zeros(len(scores))

    for i, indice in enumerate(indices):
        if r_filters[i] == 1:  # continue if this proposal is filted by nms before
            continue
        keep_index.append(indice)
        if len(keep_index) >= top_k:  # break if more than top_k
            break
        if i == (len(scores) - 1):  # break if indice is the last one
            break
        sub_indices = indices[i + 1:]
        for sub_i, sub_indice in enumerate(sub_indices):
            r_filter = Lane_IOU(proposals[indice, :], proposals[sub_indice, :], overlap)
            if r_filter: r_filters[i + 1 + sub_i] = 1
    num_to_keep = len(keep_index)
    keep_index = list(map(lambda x: x.item(), keep_index))
    return torch.tensor(keep_index), num_to_keep


def Lane_IOU(parent_box, compared_box, threshold,n_strips=71,n_offsets=72):
    '''
    calculate distance one pair of proposal lines
    return True if distance less than threshold
    '''
    start_a = (parent_box[2] * n_strips + 0.5).int()  # add 0.5 trick to make int() like round
    start_b = (compared_box[2] * n_strips + 0.5).int()
    start = torch.max(start_a, start_b)
    end_a = start_a + parent_box[5] - 1 + 0.5 - (((parent_box[5] - 1) < 0).int())
    end_b = start_b + compared_box[5] - 1 + 0.5 - (((compared_box[5] - 1) < 0).int())
    end = torch.min(torch.min(end_a, end_b), torch.tensor(n_offsets - 1))
    # end = torch.min(torch.min(end_a,end_b),torch.FloatTensor(self.n_offsets-1, device = torch.device('cpu')))
    if (end - start) < 0:
        return False
    dist = 0
    for i in range(6 + start, 6 + end.int()):
        # if i>(5+end):
        #     break
        if parent_box[i] < compared_box[i]:
            dist += compared_box[i] - parent_box[i]
        else:
            dist += parent_box[i] - compared_box[i]
    return dist < (threshold * (end - start + 1))
# end <rewrte nms for converting onnx model>

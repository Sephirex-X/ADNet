import os
import os.path as osp
import numpy as np
from .base_dataset import BaseDataset
from .registry import DATASETS
from adnet.utils.culane_metric import culane_metric,eval_predictions
import cv2
from tqdm import tqdm
import logging
import json
LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt'
} 

@DATASETS.register_module
class CULane(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        os.makedirs('cache', exist_ok=True)
        self.cache_path = 'cache/culane_{}.json'.format(split)
        self.data_infos = []        

        self.load_annotations()

    def load_annotations(self):
        if os.path.exists(self.cache_path):
            self.logger.info('Loading CULane annotations (cached)...')
            with open(self.cache_path, 'r') as cache_file:
                data = json.load(cache_file)
                self.data_infos = data['data_infos']
        else:
            self.logger.info('Loading CULane annotations...')
            
            with open(self.list_path) as list_file:
                for line in tqdm(list_file):
                    infos = self.load_annotation(line.split())
                    self.data_infos.append(infos)
            with open(self.cache_path,'w') as cache_file:
                json.dump(dict(
                    data_infos = self.data_infos
                ),cache_file)
    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line 
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        # if len(line) > 2:
        #     exist_list = [int(l) for l in line[2:]]
        #     infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 3]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.array(list(self.cfg.sample_y))[::-1] / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        result = eval_predictions(output_basedir, self.data_root, self.list_path, official=True)
        self.logger.info(result)
        return result['F1']
    
    def Lane2list_org(self,Lane_Point):
        """
        Returns a list of lanes, where each lane is a list of points (x,y)
        """
        ys = np.arange(self.cfg.ori_img_h) / self.cfg.ori_img_h
        out = []
        for lane in Lane_Point:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            out_lane = []
            for p_x,p_y in zip(lane_xs,lane_ys):
                out_lane.append([p_x,p_y])
            out.append(out_lane)
        return out

    def cal_lane_by_labels(self,predictions: list,lanes: list)->dict:
        '''calculate lane metric by dataloader's label
            input
                -  predictions & lanes
                    [
                        #per image
                        [   #per lane e.g lane1,lane2
                            [Lane[x,y],Lane[x,y]....]
                            [Lane[x,y],Lane[x,y]....]
                        ]
                        [ ...  ]
                        [ ...  ]
                    ]
            show
                - {'TP': total_tp, 
                   'FP': total_fp, 
                   'FN': total_fn, 
                   'Precision': precision, 
                   'Recall': recall, 
                   'F1': f1}
            return    
                - f1
        '''
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for pre,gt in tqdm(zip(predictions,lanes)):
            # convert from Lane to List
            pre = self.Lane2list_org(pre)
            gt = self.Lane2list_org(gt)
            tp,fp,fn,_,_ = culane_metric(pre, gt,width=30, official=True, img_shape=(590, 1640, 3))
            total_tp += tp
            total_fp += fp
            total_fn += fn
        if total_tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = float(total_tp) / (total_tp + total_fp)
            recall = float(total_tp) / (total_tp + total_fn)
            f1 = 2 * precision * recall / (precision + recall)
        self.logger.info({'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1': f1})
        return f1
import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from .registry import DATASETS
from .process import Process
from adnet.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC
from adnet.core.lane import Lane

@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None,
            cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split 
        self.processes = Process(processes, cfg)


    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes,anks, img_meta in zip(predictions[0],predictions[1], img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            gts = img_meta['no_aug_lanes']
            # img = cv2.imread(img_meta['full_img_path'])
            out_file = osp.join('vis_results', 'visualization_tusimple',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            def trans(lane, para):
                return lane.metadata[para].cpu().item()
            start_points = [(trans(lane,'start_x')*self.cfg.ori_img_w,trans(lane,'start_y')*(self.cfg.ori_img_h-self.cfg.cut_height)+self.cfg.cut_height) for lane in anks]
            anks = [lane.to_array(self.cfg) for lane in anks]

            gts = [np.array(lane) for lane in gts]
            imshow_lanes(img, lanes,anks,start_points,gts, show=False,out_file=out_file)
    #===========================================================================
    #                            online_val
    def label_to_lanes(self, label):
        cfg = self.cfg
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        lanes = []
        for l in label:
            if l[1] == 0:
                continue
            xs = l[6:] / self.img_w
            ys = self.offsets_ys / self.img_h
            start = int(round((1-l[2]) * self.n_strips))
            length = int(round(l[5]))
            xs = xs[start:start + length][::-1]
            ys = ys[start:start + length][::-1]
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            points = np.hstack((xs, ys))

            lanes.append(Lane(points=points))
        return lanes    
    #===========================================================================


    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        if self.cfg.dataset_type=='VIL':
            self.cfg.cut_height = img.shape[0] // 3
            
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:,...]
            sample.update({'mask': label})

        # if self.cfg.cut_height != 0:
        new_lanes = []
        no_aug_lanes = []
        for i in sample['lanes']:
            lanes = []
            no_aug_lane = []
            for p in i:
                lanes.append((p[0], p[1] - self.cfg.cut_height))
                no_aug_lane.append((p[0], p[1]))
            new_lanes.append(lanes)
            no_aug_lanes.append(no_aug_lane)
        sample.update({'lanes': new_lanes})
        
        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name'],
                'no_aug_lanes': no_aug_lanes}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})
        

        return sample 

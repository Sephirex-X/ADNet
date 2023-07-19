
import os
import json
import yaml
from .base_dataset import BaseDataset
from .registry import DATASETS
from tqdm import tqdm
import numpy as np
from adnet.utils.vil_metric import eval_predictions,LaneEval
from adnet.utils.vil_utils import RES_MAPPING
@DATASETS.register_module
class VIL(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None ):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        dbfile = os.path.join(data_root, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_root, 'JPEGImages')
        self.annodir = os.path.join(data_root, 'Annotations')
        self.jsondir = os.path.join(data_root,'Json')
        self.root = data_root
        self.data_infos = []
        self.folder_all_list = []
        self.sub_folder_name = []
        self.max_lane = 0
        # self.mapping = dict()
        # extract annotation information
        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']
            self.info = db
            self.videos = [info['name'] for info in db if info['set'] == split]
        self.load_annotations()
    
    def get_json_path(self,vid_path):
        json_paths = []
        for root, _, files in os.walk(vid_path):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))
        return json_paths

    def load_annotations(self):
        json_paths = []
        self.all_file_name = []
        print("Searching annotation files...")
        for vid in self.videos:
            json_paths.extend(self.get_json_path(os.path.join(self.jsondir,vid)))
        print("Found {} annotations".format(len(json_paths)))
        for json_path in tqdm(json_paths):
            with open(json_path,'r') as jfile:
                data = json.load(jfile)
            self.load_annotation(data)
            self.all_file_name.append(json_path.replace(self.jsondir+'/','')[:-9]+'.lines.txt')
        print('Max lane: {}'.format(self.max_lane))
        # print(self.mapping)
        
    def load_annotation(self,data):
        points = []
        lane_id_pool =[]
        image_path = data['info']["image_path"]
        # width,height = cv2.imread(os.path.join(self.imgdir,image_path)).shape[:2]
        mask_path = image_path.split('.')[0] + '.png'
        for lane in data['annotations']['lane']:
            # if lane['lane_id'] not in lane_id_pool:
            points.append(lane['points'])
                # lane_id_pool.append(lane['lane_id'])
        self.data_infos.append(
            dict(
                img_name = os.path.join('JPEGImages',image_path),
                # img_size = [width,height],
                img_path = os.path.join(self.imgdir,image_path),
                mask_path = os.path.join(self.annodir,mask_path),
                lanes = points
            )
        )
        sub_folder = image_path.split('/')[0]
        if sub_folder not in self.sub_folder_name:
            self.sub_folder_name.append(sub_folder)
            # self.mapping.update({sub_folder:[width,height]})
        # using index
        idx = self.sub_folder_name.index(sub_folder)
        self.folder_all_list.append(idx)
        
        
        if len(points) > self.max_lane:
            self.max_lane = len(points)
        return

    def get_prediction_string(self, pred,sub_name):
        ori_img_h,ori_img_w = RES_MAPPING[sub_name]
        ys = np.arange(ori_img_h) / ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)
        return '\n'.join(out)

    def save_tusimple_predictions(self, idx,prediction,sub_name,runtime):
        line = self.pred2tusimpleformat(idx, prediction, runtime,sub_name)
        self.tu_lines.append(line)

    
    def pred2tusimpleformat(self, idx, pred, runtime,sub_name):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred,sub_name)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)
    
    def pred2lanes(self, pred,sub_name):
        ori_img_h,ori_img_w = RES_MAPPING[sub_name]
        ys = np.arange(ori_img_h) / ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes
    def evaluate(self, predictions, output_basedir):
        print('Generating prediction output...')
        output_basedir = os.path.join(output_basedir,'preds')
        os.makedirs(output_basedir, exist_ok=True)
        for idx, pred in enumerate(tqdm(predictions)):
            sub_name = self.data_infos[idx]['img_name'].split('/')[1]
            output_dir = os.path.join(output_basedir, sub_name)
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred,sub_name)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        txt_path = os.path.join(self.data_root,'anno_txt')
        # output_basedir = '/home/xly/SPGnet/pred_txt'
        accuracy, fp, fn = LaneEval.calculate_return(output_basedir, self.jsondir)
        self.logger.info(dict(acc=accuracy,
                              fp=fp,
                              fn=fn))
        result = eval_predictions(output_basedir, txt_path, self.all_file_name, official=False,iou_thresholds=[0.5])
        # 
        self.logger.info(result)
        
        return result[0.5]['F1']
        # return accuracy

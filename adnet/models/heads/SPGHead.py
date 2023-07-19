import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from adnet.core.lane import Lane
from adnet.models.losses.focal_loss import FocalLoss
from adnet.models.losses.Glineiou_loss import Gliou_loss
from adnet.models.losses.lineiou_loss import liou_loss
from adnet.ops import nms
from ..registry import HEADS
from adnet.models.utils.SPGU import SPGU
from adnet.models.utils.ALAU import ALAU
from adnet.models.utils.dynamic_assign import assign
from adnet.models.losses.GfocalLoss import GFocalLoss
from adnet.models.losses.RegL1ThetaLoss import RegL1ThetaLoss
@HEADS.register_module
class SPGHead(nn.Module):
    def __init__(self,
                 S=72,
                 img_width=640,
                 img_height=360,
                 anchor_feat_channels=64,
                 start_points_num=100,
                 cfg=None):
        super(SPGHead, self).__init__()
        self.cfg = cfg
        assert cfg.hm_down_scale in cfg.fpn_down_scale
        self.hm_fmap_id = cfg.fpn_down_scale.index(cfg.hm_down_scale)
        self.ALAU = ALAU(anchor_feat_channels,anchor_feat_channels,kernel_size=(3,3),deform_groups=2,cfg=self.cfg)
        self.num_anchor = start_points_num
        self.img_w = img_width
        self.img_h = img_height
        self.n_strips = S - 1
        self.n_offsets = S
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        self.SPGU = SPGU(
            heads=dict(hm=1,shape = 1),
            channels_in=anchor_feat_channels,
            final_kernel=1,
            head_conv=anchor_feat_channels)
        self.conv_downsample = ConvModule(self.anchor_feat_channels,
                            self.anchor_feat_channels, (9, 1),
                            padding=(4, 0),
                            bias=False,
                            norm_cfg=dict(type='BN'))
        self.downsample_layer = nn.Linear(self.n_offsets*self.anchor_feat_channels,self.anchor_feat_channels)
        self.fc_norm = nn.LayerNorm(self.anchor_feat_channels)
        self.cls_layer = nn.Linear(self.anchor_feat_channels, 2)
        self.reg_layer = nn.Linear(self.anchor_feat_channels, self.n_offsets + 1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)
        self.initialize_layer(self.conv_downsample)
        self.anchor_ys = self.anchor_ys.cuda()
        # * debug
        self.hack = False
    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        '''

        batch_size = batch_features.shape[0]

        # prior_xs = prior_xs.repeat((batch_size,1)).view(batch_size, num_priors, -1, 1)/self.img_w
        prior_xs /= self.cfg.img_w
        prior_xs = prior_xs.reshape(batch_size, num_priors, -1, 1)
        prior_ys = self.anchor_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1).float()
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        feature = feature.reshape(batch_size * num_priors,
                                  self.anchor_feat_channels, self.n_offsets,
                                  1)
        roi = self.conv_downsample(feature)
        roi = roi.contiguous().view(batch_size*num_priors,-1)
        roi = F.relu(self.fc_norm(self.downsample_layer(roi)))
        roi = roi.view(batch_size,num_priors,-1)
        return roi

    def decode_theta_xy(self,heat_xy,heat_theta,topk):
        heat_nms = heat_xy[0]
        all_idx = heat_nms.nonzero()
        heat_nms_scores = heat_nms.flatten()
        _,top_k_socre_idx = torch.topk(heat_nms_scores,k=topk,largest=True)
        xy_anchor_idx = all_idx[top_k_socre_idx]
        xy_starts = torch.zeros_like(xy_anchor_idx,dtype=torch.float64)
        xy_starts[:,0] = xy_anchor_idx[:,1]*self.cfg.hm_down_scale/self.cfg.img_w
        xy_starts[:,1] = xy_anchor_idx[:,0]*self.cfg.hm_down_scale / self.cfg.img_h
        thetas = heat_theta.flatten()[top_k_socre_idx]/180        
        return xy_starts,thetas        
    def forward(self, x, **kwargs):
        param = self.cfg.train_parameters if self.training else self.cfg.test_parameters
        conf_threshold=param.conf_threshold
        nms_thres=param.nms_thres
        nms_topk=param.nms_topk
        bs = x[-1].shape[0]
        device = x[-1].device
        # x16
        f_hm = x[self.hm_fmap_id]
        z = self.SPGU(f_hm)
        # hm (bs,1,hm_h,hm_w)
        hm = z['hm']
        shape_hm = z['shape']
        # param = z['parameter']
        theta_hm = shape_hm[:,0]
        # x32
        # batch_features = x[-1]
        batch_features,offset = self.ALAU(x[-1],torch.hstack([hm,shape_hm]))
        # batch_features,offset = self.ALAU(x[-1],shape_hm)

        self.anchors = torch.zeros((bs,self.num_anchor,2 + 2 + 2 + self.n_offsets),device = device)
        hms = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        # theta_hms = torch.clamp(theta_hm.clone().sigmoid(), min=1e-4, max=1 - 1e-4)
        theta_hms = theta_hm
        
        for bs_id,(hm_c,theta_hm_c) in enumerate(zip(hms,theta_hms)):
            with torch.no_grad():
                ct_points,thetas = self.decode_theta_xy(hm_c,theta_hm_c,topk=self.num_anchor)
                assert len(ct_points) == self.num_anchor
            anchors_per_batch = self.generate_anchors_in_one(starts=ct_points,angles=thetas)
            anchors_per_batch = anchors_per_batch.to(device)
            self.anchors[bs_id] = anchors_per_batch


        assert self.num_anchor == len(anchors_per_batch)
        batch_anchor_features = self.pool_prior_features(batch_features,self.num_anchor,self.anchors[:,:,6:].clone())
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels)
         # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(bs, -1, cls_logits.shape[1])
        reg = reg.reshape(bs, -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 6 + self.n_offsets), device=device)
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 5:] += reg
        
        # Apply nms
        proposals_list = self.nms(reg_proposals, nms_thres, nms_topk, conf_threshold)  
        if self.cfg.dataset_type == 'VIL' and not self.training:
            from adnet.utils.vil_utils import RES_MAPPING
            sub_name = kwargs['batch']['meta'][0]['img_name'].split('/')[1]
            self.cfg.ori_img_h,self.cfg.ori_img_w = RES_MAPPING[sub_name]
            self.cfg.cut_height = self.cfg.ori_img_h // 3
        output =dict(
                hm = hm,
                shape_hm = shape_hm,
                proposals_list = proposals_list,
                anchor_info = list(self.anchors)
        )         
        return output 

    def nms(self, batch_proposals, nms_thres, nms_topk, conf_threshold):
        proposals_list = []
        # anchors_local = self.anchors
        for proposals in batch_proposals:
            softmax = nn.Softmax(dim=1)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                keep = None
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]],None, None, None))
                    continue                
                if conf_threshold is not None:
                    # * CUDA error invalid argument will occur
                    # try:
                    scores = softmax(proposals[:, :2])[:, 1]
                    # scores = proposals[:,1].cpu().numpy()
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    if proposals.shape[0] == 0:
                        proposals_list.append(proposals[[]])
                        continue    
                    scores = scores[above_threshold]
                    # keep sp generated by hm and swap x and y
                    start_points_hm = proposals[:,2:4]
                    start_points_hm = start_points_hm[:,[1,0]]
                    # end
                    proposals = self.relocate_xy(proposals)
                    keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                    keep = keep[:num_to_keep]
                    # except:
                    #     keep = None
            if keep is not None:
                proposals = proposals[keep]
                start_points_hm = start_points_hm[keep]
                if self.cfg.dataset_type == 'VIL' and not self.training:
                    from adnet.utils.vil_utils import relocate2mid
                    proposals_np = relocate2mid(proposals.cpu().numpy(),start_points_hm.cpu().numpy()*np.array([self.cfg.img_w,self.cfg.img_h]),cfg=self.cfg)
                    proposals = torch.tensor(proposals_np).to(scores.device)
                proposals_list.append(proposals)
            else:
                proposals_list.append(proposals)

        return proposals_list

    def loss(self, output, batch):
        hm_gt = batch['gt_hm']
        shape_hm_gt = batch['shape_hm']
        shape_hm_mask = batch['shape_hm_mask']
        shape_hm = output['shape_hm']
        hm = output['hm'] 
        hm_focalloss = GFocalLoss()
        theta_regloss = RegL1ThetaLoss()
        hm_loss = 0
        theta_loss = 0
        hm = torch.clamp(hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        targets = batch['lane_line']
        imgs = batch['img']
        proposals_list = output['proposals_list'] 
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        l1loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        length_loss = 0
        valid_imgs = len(targets)
        for proposals, target,hm_p,shape_hm_p,hm_g,shape_hm_g,shape_hm_m in zip(proposals_list, targets,hm,shape_hm,hm_gt,shape_hm_gt,shape_hm_mask):

            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            # hm
            hm_loss += hm_focalloss((hm_p).unsqueeze(0),(hm_g).unsqueeze(0))
            theta_loss += theta_regloss(shape_hm_p.unsqueeze(0),shape_hm_g.unsqueeze(0),shape_hm_m.unsqueeze(0))
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                matched_row_inds, matched_col_inds = assign(
                    proposals, target, self.cfg.img_w, self.cfg.img_h)
            # Get classification targets
            cls_target = proposals.new_zeros(proposals.shape[0]).long()
            cls_target[matched_row_inds] = 1.
            cls_pred = proposals[:, :2]

            # Regression targets
            reg_pred = proposals[matched_row_inds, 6:]
            reg_target = target[matched_col_inds, 6:].clone()
            #  length 
            length_pred = proposals[matched_row_inds, 5]
            length_target = target[matched_col_inds, 5].clone()
            # Loss calc
            reg_loss += Gliou_loss(reg_pred, reg_target,img_w=self.cfg.img_w)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / target.shape[0]
            length_loss += l1loss(length_pred,length_target)
        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs
        length_loss /= valid_imgs
        hm_loss /= valid_imgs
        theta_loss /= valid_imgs
        # cls reg l1 10:1
        # cls reg liou 1:6
        regw = 6
        hmw = 2
        thetalossw = 3
        cls_loss_w = 1        
        if self.cfg.haskey('regw'):
            regw = self.cfg.regw
        if self.cfg.haskey('hmw'):
            hmw = self.cfg.hmw
        if self.cfg.haskey('thetalossw'):
            thetalossw = self.cfg.thetalossw
        if self.cfg.haskey('cls_loss_w'):
            cls_loss_w = self.cfg.cls_loss_w
        loss = cls_loss*cls_loss_w + reg_loss * regw + hm_loss * hmw + theta_loss*thetalossw + length_loss
        return {'loss': loss,
                'loss_stats':
                    {
                        'cls_loss': cls_loss, 
                        'reg_loss': reg_loss,
                        'hm_loss':hm_loss,
                        'theta_loss':theta_loss,
                        'length_loss': length_loss
                    }
                }
    
    def find_first_nonezero(self,x):
        index = torch.arange(x.shape[1]).unsqueeze(0).repeat((x.shape[0],1))
        index[x==0] = x.shape[1]
        return torch.min(index,dim=1)[0]
        
    def relocate_xy(self,proposals):
        #================================================
        #        put start x and y into axis        
        #================================================
        # proposals[bs *n, 78]
        new_proposals = proposals.clone()
        with torch.no_grad():
            proposals_matched_positive_reg = proposals[:,6:].clone()
            invali_mask = (proposals_matched_positive_reg<0) | (proposals_matched_positive_reg>self.cfg.img_w)
            # puts invalid points into zero
            proposals_matched_positive_reg = torch.where(~invali_mask,proposals_matched_positive_reg,torch.tensor([0],dtype=torch.float,device=proposals_matched_positive_reg.device))
            # a = torch.tensor([[0,1,2,3],[3,2,1,0],[0,0,3,4]])
            idx = self.find_first_nonezero(proposals_matched_positive_reg).to(proposals_matched_positive_reg.device)
            x_s = proposals_matched_positive_reg[torch.arange(proposals_matched_positive_reg.shape[0]),idx-1] / self.cfg.img_w
            y_s = idx / self.n_strips
            start_xy = torch.hstack((y_s.view(-1,1),x_s.view(-1,1)))
            none_zero_idx = y_s.nonzero()
            start_xy[none_zero_idx,1] = torch.round(start_xy[none_zero_idx,1])
            new_proposals[:,2:4] = start_xy
            return new_proposals

    def limit_to_length(self,proposals):
        with torch.no_grad():   
            proposals_matched_positive_reg = proposals[:,6:].clone()
            # points that smaller than zero and larger than img width is invalid
            
            invali_mask = (proposals_matched_positive_reg<0) | (proposals_matched_positive_reg>self.cfg.img_w)
            # puts invalid points into zero
            proposals_matched_positive_reg = torch.where(~invali_mask,proposals_matched_positive_reg,torch.tensor([0],dtype=torch.float,device=proposals_matched_positive_reg.device))
            proposals_length = proposals[:,5]
            for i in range(len(proposals_matched_positive_reg)):
                try:
                    idx = torch.argwhere(proposals_matched_positive_reg[i])[0].to(proposals_length.device)
                    end_idx = (idx + proposals_length[i]).long()
                    proposals_matched_positive_reg[i,end_idx:] = 0 
                except:
                    continue
            proposals_matched_positive_reg = torch.where(proposals_matched_positive_reg!=0,proposals_matched_positive_reg,torch.tensor([-1000],dtype=torch.float,device=proposals_matched_positive_reg.device))
        proposals[:,6:] = proposals_matched_positive_reg
        return proposals
    

    def generate_anchors_in_one(self,starts,angles):
        def trans(input):
            return input.unsqueeze(-1).repeat(1,72)
        anchors = torch.zeros((len(starts), 2 + 2 + 2 + self.n_offsets))
        # starts = starts.reshape(-1,2)
        # angles = angles.reshape(-1,1)
        anchor_ys = self.anchor_ys.unsqueeze(0).repeat(len(starts),1)
        anchors[:,4] = angles.clone()
        angle = (1-angles) * 180
        angle = angle * math.pi / 180
        start_x,start_y = starts[:,0],starts[:,1]
        anchors[:,2] = start_y
        anchors[:,3] = start_x
        anchors[:,6:] = ((anchor_ys - trans(start_y)) * self.cfg.img_h / torch.tan(trans(angle)+1e-4)) + trans(start_x) * self.cfg.img_w
        return anchors                

    def generate_anchors(self,starts,angles):
        anchors = torch.zeros((len(starts), 2 + 2 + 2 + self.n_offsets))
        for i, (start,angle) in enumerate(zip(starts,angles)):
            anchors[i] = self.generate_anchor(start,angle)
        return anchors

    def generate_anchor(self, start, angle, cut=False):
        #===========================================================================
        #  *                                 INFO
        #    the start_y,start_x,angle is unified 0~1
        #    angle is between the ray and x-axis in clockwise direction e.g 30 60
        #    
        # 
        #===========================================================================
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 2 + self.fmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 2 + self.n_offsets)
        anchor[4] = angle.clone()
        angle = (1-angle) * 180
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = start_y
        anchor[3] = start_x
        anchor[6:] = ((anchor_ys - start_y) * self.cfg.img_h / math.tan(angle)) + start_x * self.cfg.img_w
        return anchor

    # per img
    def draw_anchors(self,img, img_w, img_h,anchors ,k=None,thickness=4,color = None):
        from adnet.utils.visualization import _COLORS
        with torch.no_grad():
            base_ys = self.anchor_ys.cpu().numpy()
            # img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            i = -1
            for color_id,anchor in enumerate(anchors):
                if color is None:
                    line_color = (_COLORS[color_id] * 255 * 0.7).astype(np.uint8).tolist()
                else:
                    line_color = color
                i += 1
                if k is not None and i != k:
                    continue
                anchor = anchor.numpy()
                xs = anchor[6:]
                ys = (base_ys) * img_h
                points = np.vstack((xs, ys)).T.round().astype(int)
                try:
                    for p_curr, p_next in zip(points[:-1], points[1:]):
                        if (p_next[0] < 0) or (p_curr[0]<0):
                            continue
                        img = cv2.line(img, tuple(p_curr), tuple(p_next), color=line_color, thickness=thickness)
                except:
                    continue
        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[6:] / self.cfg.img_w
            length = int(round(lane[5].item()))
            # invali_mask = (lane_xs<0) | (lane_xs>1)
            # proposals_matched_positive_reg = torch.where(~invali_mask,lane_xs,torch.tensor([0],dtype=torch.float,device=lane_xs.device))
            
            # idx = torch.argwhere(proposals_matched_positive_reg)[0].to(proposals_matched_positive_reg.device)
            # start = idx * self.n_strips
            # end = (idx + length).long()      
            
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)            
            
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                       self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, as_lanes=True):
        proposals_list = output['proposals_list']
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 5] = torch.round(proposals[:, 5])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded
    def get_lanes_temp(self, output, as_lanes=True):
        proposals_list = output['anchor_info']
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 5] = torch.round(proposals[:, 5])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred_temp(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded
    def proposals_to_pred_temp(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[6:] / self.cfg.img_w
            
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)            
            
            end = len(self.anchor_ys) - 1
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                       self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes
import torch.nn as nn
import torch

from adnet.models.registry import NETS
from ..registry import build_backbones, build_heads, build_necks


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
    
    def get_lanes(self, output):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'])
        fea_org = fea

        if self.neck: 
            fea = self.neck(fea)
        
        if self.training:
            out = self.heads(fea, batch=batch,fea_org=fea_org)
            output.update(self.heads.loss(out, batch))
        else:
            output = self.heads(fea,batch=batch,fea_org=fea_org)

        return output

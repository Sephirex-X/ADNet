import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import torch
from adnet.utils.config import Config
from adnet.models.registry import build_net
import mmcv
import cv2
def build_your_net(cfg):
    net = build_net(cfg)
# build your network according to code
    return net
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = 'cuda'
    repeat_time = 2000
    bs = 1
    path = 'configs/adnet/tusimple/resnet18_tusimple.py'
    cfg = Config.fromfile(path)
    # cfg is used to build model
    net = build_your_net(cfg).to(device)
    net.eval()
    # input shape is (bs,channels,img_h,img_w)
    # here we use bs=1 channels=3 as fixed parameter, img_h,img_w should be refered to model's config file accordingly
    input = torch.zeros((bs,3,320,800),device=device)
    data = {'img':input}
    for i in range(200):
        out = net(data)    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in mmcv.track_iter_progress(range(repeat_time)):
        out = net(data)
    torch.cuda.synchronize()
    end = time.perf_counter()
    fps = 1/((end-start) / (repeat_time*bs))
    print(fps)
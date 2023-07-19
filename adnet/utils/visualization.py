import cv2
import os
import os.path as osp
def draw_lanes(img,lanes,color=None,thickness=15):
    for color_id,lane in enumerate(lanes):
        if color is None:
            line_color = (_COLORS[color_id] * 255 * 0.7).astype(np.uint8).tolist()
        else:
            line_color = color
        lane = lane.astype(int)
        for p_curr, p_next in zip(lane[:-1], lane[1:]):
            if (p_next[0] < 0) or (p_curr[0]<0):
                continue
            img = cv2.line(img, tuple(p_curr), tuple(p_next), color=line_color, thickness=thickness)   
    return img 
def imshow_lanes(img, lanes,anks,start_points,gt_lanes, show=False, out_file=None,size=(800,320)):
    # draw_preds
    raw = img.copy()
    blank = np.zeros_like(raw)
    # blank[...,0] = 84
    # blank[...,1] = 1
    # blank[...,2] = 68
    img_pd=draw_lanes(blank.copy(),lanes)
    img_gt = draw_lanes(blank.copy(),gt_lanes,color=(0,255,0))
    img_anks = draw_lanes(blank.copy(),anks,color=(255,255,0),thickness=1)
    for points in start_points:
        cv2.circle(img=img_anks,center=(int(points[0]),int(points[1])),radius=4,color=(200,0,200),thickness=4) 
    raw = cv2.resize(raw,size)
    img_pd = cv2.resize(img_pd,size)
    img_gt = cv2.resize(img_gt,size)
    img_anks = cv2.resize(img_anks,size)

    bar = np.ones((10,size[0],3))*255
    img_cat = np.concatenate((raw,bar,img_gt,bar,img_anks,bar,img_pd),axis=0)
        # for x, y in lane:
        #     if x <= 0 or y <= 0:
        #         continue
        #     x, y = int(x), int(y)
        #     cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

    if show:
        # cv2.imwrite('view.jpg', img_pd)
        print("pass")

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img_cat)

import numpy as np
# _COLORS = np.array(
#     [
#         0.57, 0.69, 0.30,
#          0.89, 0.88, 0.57,
#          0.76, 0.49, 0.58,
#          0.47, 0.76, 0.81,
#          0.21, 0.21, 0.35,
#          0.28, 0.57, 0.54,
#     ]
# ).astype(np.float32).reshape(-1, 3)
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
# basic setting
img_w = 800
img_h = 320
# inference setting
max_lanes = 6

train_parameters = dict(
    conf_threshold=None,
    nms_thres=45.,
    nms_topk=max_lanes
)
test_parameters = dict(
    conf_threshold=0.3,
    nms_thres=45,
    nms_topk=max_lanes
)

# dataset setting
sample_y=range(1080, 134, -1)
hm_down_scale = 8
keys = ['img', 'lane_line','gt_hm','shape_hm','shape_hm_mask']
train_process = [
    dict(
        type='GenerateLanePts',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],

    ),
    dict(type='CollectHm',
    down_scale=hm_down_scale,
    hm_down_scale=hm_down_scale,
    max_mask_sample=5,
    line_width=3,
    # 6
    radius=6,
    theta_thr = 0.2,
    keys=keys,
    meta_keys=['gt_points']
    ),    
    dict(type='ToTensor', keys=keys),
]

val_process = [
        dict(
        type='GenerateLanePts',
        training = False,
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],

    ),
    # dict(type='CollectHm',
    # down_scale=hm_down_scale,
    # hm_down_scale=hm_down_scale,
    # max_mask_sample=5,
    # line_width=3,
    # radius=6,
    # theta_thr = 0.2,
    # keys=keys,
    # meta_keys=['gt_points']
    # ),      
    dict(type='ToTensor', keys=['img','lane_line']),
]

dataset_path = './data/VIL100'
dataset_type = 'VIL'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)
workers = 4
log_interval = 100
seed=0
lr_update_by_epoch = False
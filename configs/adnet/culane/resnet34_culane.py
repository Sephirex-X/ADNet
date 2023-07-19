_base_ = ['../_base_/dataset_culane.py',
            '../_base_/base_res34.py']
# basic setting
img_w = 800
img_h = 320
# work_dir = './test_dir/culane/cu_res34_noSA'
# network setting
fpn_down_scale = [8,16,32]
anchors_num = 300
# neck = dict(type='FPN')
heads = dict(type='SPGHead',
        img_width = img_w,
        img_height = img_h,
        start_points_num=anchors_num)
# train setting
regw = 6
hmw = 2
thetalossw = 3
cls_loss_w = 6
type = 'AdamW'
lr = 0.0007

epochs = 15
batch_size = 45
batch_size_test = 350

dynamic_after = 10
eval_ep = 3
save_ep = 1
do_mask = False

optimizer = dict(
  type = type,
  lr = lr,
)

total_iter = (88880 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)



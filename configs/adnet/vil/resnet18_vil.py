_base_ = ['../_base_/dataset_vil.py',
            '../_base_/base_res18.py']
# basic setting
img_w = 800
img_h = 320
# work_dir = './test_dir/vil/vil_res18_r3_app'
# network setting
fpn_down_scale = [8,16,32]
anchors_num = 75

heads = dict(type='SPGHead',
        img_width = img_w,
        img_height = img_h,
        start_points_num=anchors_num)

# train setting
# regw = 10
# hmw = 5
# thetalossw = 1
# cls_loss_w = 10
regw = 10
hmw = 10
thetalossw = 1
cls_loss_w = 10
type = 'Adam'
lr = 0.0012

epochs = 80
batch_size = 45
batch_size_test = 100

dynamic_after = 60
eval_ep = 10
save_ep = 1
do_mask = False

optimizer = dict(
  type = type,
  lr = lr,
)

total_iter = (8000 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)



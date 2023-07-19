resnet = 'resnet18'
in_channels= [128,256,512]
anchor_feat_channels = 64
num_points = 72
net = dict(
    type='Detector',
)
backbone = dict(
    type='ResNetWrapper',
    resnet=resnet,
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

neck = dict(type='SA_FPN',
            in_channels=in_channels,
            out_channels=anchor_feat_channels,
            num_outs=len(in_channels))
heads = dict(type='SPGHead',
        S = num_points,
        anchor_feat_channels = anchor_feat_channels
        )
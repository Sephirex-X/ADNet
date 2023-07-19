in_channels= [128,320,512]
anchor_feat_channels = 64
num_points = 72
net = dict(
    type='Detector',
)
backbone = dict(
    type='MSCANWrapper',
    mscan='mscan_s',
    pretrained=True,
)
neck = dict(type='SA_FPN',
            in_channels=in_channels,
            out_channels=anchor_feat_channels,
            num_outs=len(in_channels))
heads = dict(type='SPGHead',
        S = num_points,
        anchor_feat_channels = anchor_feat_channels
        )
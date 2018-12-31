DetectionCfg = {
    'feature_maps': [[80, 45], [40, 23], [20, 12], [10, 6], [5, 3]],

    'input_dim': [640, 360],

    'anchors': [[32], [64], [128], [256], [512]],

    'aspect_ratios': [1/2., 1/1., 2/1.],

    'scale_ratios': [1., pow(2, 1/3.), pow(2, 2/3.)],

    'variance': [0.1, 0.2],

    'num_class': 1,

    'iou_thr': 0.5,

    'conf_thr': 0.6,

    'neg_pos': 3,

    'clip': True,

    'lr_steps': [5, 15, 30, 60, 120, 240]
}

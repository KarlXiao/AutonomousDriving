DetectionCfg = {
    'feature_maps': [[20, 12], [10, 6], [5, 3]],

    'input_dim': [640, 360],

    'anchors': [[50, 100], [150, 200], [250, 300]],

    'aspect_ratios': [1/1., 2/1.],

    'scale_ratios': [1., pow(2, 1/3.)],

    'variance': [0.1, 0.2],

    'num_class': 1,

    'iou_thr': 0.4,

    'conf_thr': 0.5,

    'neg_pos': 3,

    'clip': True,
}

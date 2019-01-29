# AutonomousDriving


Autonomous driving perception including segmentation and detection.

```
.
├── chcekpoint
|   └── *
├── config
├── core
|   ├── models
|   └── *.py
├── data
|   └── BDD100K
├── runs
|   └── *
├── tool
|   └── *.py
├── *.py
├── LICENSE
└── README.md
```

Dataset used in this project is [BDD100K](http://bdd-data.berkeley.edu/index.html), check [license](http://bdd-data.berkeley.edu/portal.html#download)

## Model

Use GAN loss as segmentation loss

## Train

![cls_loss.png](./images/cls_loss.png)
![dis_loss.png](./images/dis_loss.png)
![loc_loss.png](./images/loc_loss.png)
![seg_loss.png](./images/seg_loss.png)

## Result

![0.jpg](./images/0.jpg)
![1.jpg](./images/1.jpg)
![2.jpg](./images/2.jpg)
![3.jpg](./images/3.jpg)
![4.jpg](./images/4.jpg)
![5.jpg](./images/5.jpg)
![6.jpg](./images/6.jpg)
![7.jpg](./images/7.jpg)
![8.jpg](./images/8.jpg)
![9.jpg](./images/9.jpg)

## License

Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes.

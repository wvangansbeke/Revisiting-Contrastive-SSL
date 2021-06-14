# VOC SVM

We evaluate the representations under the linear evaluation protocol on PASCAL VOC. Since the images can contain multiple classes, we train a SVM for every class. The mAP metric is reported. To evaluate a pretrained MoCo model, run the code as follows:
```shell
python main.py --data $DATA_DIR --pretrained-weights $MOCO_PRETRAINED_WEIGHTS
```

The data will be downloaded automatically when running the code for the first time. 

NOTE: (June 10th) the server with the publicly available dataset is offline, we will double check this section once it is back online.

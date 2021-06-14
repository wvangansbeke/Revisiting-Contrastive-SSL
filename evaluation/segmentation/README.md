# Semantic Segmentation

We transfer the representations to tackle the semantic segmentation task on PASCAL VOC and Cityscapes.

## Setup
The following files need to be adapted in order to run the code on your own machine:

- Set the file paths for the datasets in `utils/mypath.py`, e.g. `/path/to/datasets/`. More information can be found in the datasets section.
- Specify where the results should be saved in `configs/env.yml`. 
- Set the path to the MoCo pretrained weights in `configs/cityscapes.yml` or `configs/pascal_voc.yml`.  


## Datasets
- __PASCAL VOC__: We use the data preprocessed by [Vandenhende et al.](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch). The dataset will be downloaded automatically on the first run.
- __Cityscapes__: The Cityscapes dataset should be downloaded from the official source. We use the fine annotations for the train/val splits.

## Usage
After completing the setup, you can run the code as follows:
```shell
python single_gpu.py --config_env configs/env.yml --config_exp configs/$EXPERIMENT.yml
```

We also support multi-gpu training:
```shell
python multi_gpu.py --config_env configs/env.yml --config_exp configs/$EXPERIMENT.yml --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
```

The `--run-idx` flag facilitates multiple runs of the same experiment. The paper reports the average of five runs. For PASCAL VOC, we use the single-gpu training script, while for Cityscapes we performed multi-gpu training on 2 x 1080ti's.

## License
This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).

## Acknoledgements
The authors acknowledge support by Toyota via the TRACE project and MACCHINA (KULeuven, C14/18/065).

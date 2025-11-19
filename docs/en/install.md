# Installation

**Step 1.** Install torch==2.7.0, Python=3.10.12

```
conda create -n opentad python=3.10.12
source activate opentad
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

**Step 2.** Install mmaction2 for end-to-end training
```
pip install openmim
pip install mmcv==2.0.1 -use-pep517
pip install mmaction2==1.1.0 -use-pep517
```

**Step 3.** Install other requirements

git clone and cd into project
```
pip install -r requirements.txt
```

The code is tested with Python 3.10.12, torch==2.7.0, CUDA 12.8, other versions might also work.

**Step 4.** download the ckpt.

Put this checkpoint into the /pretrain/ folder. The file uncompressed-thumos-videomae_small.pth is required, while others is optional.

 [uncompressed-moel](https://pan.baidu.com/s/1Z2fkI_24vX6hchsJlQhZnA?pwd=tpbd)

[compressed-model-anet](https://pan.baidu.com/s/1xFP1lL5-AxwiwYPGzE34ng?pwd=tpbd)

**Step 5.** Prepare the data.

Note: For this project, you only need to download the raw videos. The pre-extracted features are not required.

| Dataset | Description | Change "data_path" in config file |
| :--------------------------------------------------------- | :-------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| [THUMOS14](/tools/prepare_data/thumos/README.md)           | Consists of 413 videos with temporal annotations.                                             | configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py |
| (optional) [ActivityNet](/tools/prepare_data/activitynet/README.md)   | A Large-Scale Video Benchmark for Human Activity Understanding with 19,994 videos.            | configs/_base_/datasets/activitynet-1.3/e2e_resize_768_1x224x224.py |

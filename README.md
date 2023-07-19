# ADNet: Lane Shape Prediction via Anchor Decomposition 

Pytorch implementation of the paper "ADNet: Lane Shape Prediction via Anchor Decomposition " (ICCV2023 Acceptance)
## Introduction

## Environment setup

1. create conda environment if you using conda

```bash
conda create -n ADNet && conda activate ADNet
```

2. install pytorch and Shapely

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

```bash
conda install Shapely==1.7.0
```

3. building independencies

```bash
pip install -r requirements.txt
```

adding `-i https://pypi.tuna.tsinghua.edu.cn/simple` if you locate in China mainland.

4. setup everything

```bash
python setup.py build develop
```

## Benchmarks setup

1. create folder under root path using

```bash
mkdir data
```

2. organize structure like these:

```bash
data/
├── CULane -> /mnt/data/xly/CULane/
├── tusimple -> /mnt/data/xly/tusimple/
└── VIL100 -> /mnt/data/xly/VIL100/
```

under each folder, you may see structure like this:

### CULane

```
/mnt/data/xly/CULane/
├── driver_100_30frame
├── driver_161_90frame
├── driver_182_30frame
├── driver_193_90frame
├── driver_23_30frame
├── driver_37_30frame
├── laneseg_label_w16
└── list
```

### Tusimple

```
/mnt/data/xly/tusimple/
├── clips
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
├── test_label.json
└── test_tasks_0627.json

```

### VIL-100

```bash
/mnt/data/xly/VIL100/
├── Annotations
├── anno_txt
├── data
├── JPEGImages
└── Json
```

## Inferencing

1. You can inferencing model using: 

```bash
python main.py {configs you want to use} --work_dirs {your folder} --load_from {your checkpoint path} --validate --gpus {device id}
```

for example:

```bash
python main.py configs/adnet/tusimple/resnet18_tusimple.py --work_dirs test --load_from best_ckpt/tusimple/res18/best.pth --validate --gpus 3
```

If you don't assign `--work_dirs`, it will create folder named  `work_dirs` under root path by default.

2. By adding `--view` you can see visualization results under folder `vis_results`, under root path.

3. You can test fps using:

```bash
python tools/fps_test.py
```

## Training (only support single gpu)

1. Simply using:

```bash
python main.py {configs you want to use} --work_dirs {your folder} --gpus {device id}
```

2. You can resume from your last checkpoint by: 

```bash
python main.py {configs you want to use} --work_dirs {your folder} --load_from {your checkpoint path} --gpus {device id}
```

3. Other hyperparameters can be changed within config files.

4. Your can check training procedure using tensorboard module:

   ```bash
   tensorboard --logdir {your folder} --host=0.0.0.0 --port=1234
   ```

   
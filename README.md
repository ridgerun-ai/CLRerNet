# CLRerNet Official Implementation

The official implementation of [our paper](https://arxiv.org/abs/2305.08366) "CLRerNet: Improving Confidence of Lane Detection with LaneIoU", by Hiroto Honda and Yusuke Uchida.

## Installation

1. Install the project:
```bash
pip install torch==1.12.1 torchvision==0.13.1
pip3 install -r requirements.txt
pip3 install .
```

2. Install mmcv:
```bash
mim install mmcv-full==1.7.0
```

3. Install nms:
```bash
./install_nms.sh
```

4. Download the inference model:
```shell
./download_model.sh
```

## Inference

Run the following command to detect the lanes from the image and visualize them:
```bash
python demo/image_demo.py demo/demo.jpg configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file=result.png
```

Run the following command to detect the lanes from the video and visualize them:
```bash
python demo/video_demo.py demo/demo.mp4 configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file=result.mp4
```
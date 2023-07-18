# Requirements
tested on
```
python==3.8.13
torch==1.11.0
tensorboardX
torchsummary
```

# Usage
Run `main.py` with several arguments
* `--mode` : 'test' or 'train'
* `--download` : dataset을 다운로드할지 여부 결정
* `--dataset_dir` : dataset 다운로드 위치 지정
* `--output_dir` : epoch 마다 결과를 저장할 위치 지정
* `--checkpoint` : test mode에서 사용할 checkpoint 파일 지정

- - -

# Dataset: STL10
more information, visit [torchvision website](https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html) or [STL-10 website](https://cs.stanford.edu/~acoates/stl10/)

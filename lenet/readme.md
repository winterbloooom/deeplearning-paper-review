# Goal
* Make LeNet-5 Model using PyTorch (LeNet-5 모델을 파이토치를 사용해 구현해본다.)
* Train, evaluate, and test with FashionMNIST dataset. (FashionMNIST 데이터셋을 사용해 모델을 훈련시키고, 평가한다.)
* Use various methods to improve accuracy and draw training loss graphs. (다양한 방법을 사용해 모델의 정확도를 향상시켜보고, Loss 그래프를 그려본다.)
* Get an accuracy score for the model. (모델의 정확도 점수를 산출한다.)
* With 5 samples, predict the target(answer, label) and draw the image with the prediction result (5개의 샘플에 대해 정답을 예측해보고, 그 이미지와 예측 결과를 함께 출력한다.)

> This is an assignment for the Course - Programmers Autonomous Driving Dev. Course (3rd). (이 내용은 프로그래머스 자율주행 데브코스 3기의 과제입니다.)

- - -

# Usage
1. Activate Conda Virtual Environment. (For me, `activate pytorch_py38` in VS Code Command Prompt)
2. Run `main.py` with several arguments. (e.g. `python main.py --mode test --download 1 --output_dir ./output --checkpoint ./output/model_epoch3.pt --dataset FashionMNIST`)
    * `--mode` for choosing the mode of model - train, eval(evaluation), test. In 'train' mode, the model train the prepared dataset. 'eval' mode let the model calculate the accuracy of the prediction of the model by comparing the label of each image. In 'test' mode, the test image is shown with its predicted results.
    * `--download` : specify whether you download the dataset or not. (Bool)
    * `--output_dir` : specify where the output of the model(parameters, etc) is stored.
    * `--checkpoint` : specify which `.pt` file you will use during evaluation or test mode
    * `--dataset` : specify the datasets you would use - MNIST or FashionMNIST

- - -

# Dataset: FashionMNIST
* 28 X 28 size Grayscale images
* 10 classes

|label|description|label|description|
|---|---|---|---|
|0| T-shirt/top|1| Trouser|
|2| Pullover|3| Dress|
|4| Coat|5| Sandal|
|6| Shirt| 7| Sneaker|
|8| Bag|9| Ankle boot|

* Can use MNIST dataset either, using `--dataset` argument
```py
parser.add_argument('--dataset',
                    dest = 'dataset',
                    help = 'dataset to train model',
                    default = None,
                    type = str) # MNIST, FashionMNIST
```

- - -

# Model: LeNet-5
## Basic Setup
* Layers : Convolution 0 → Pooling 0 → Convolution 1 → Pooling 1 → Convolution 2 → Fully Connedted Layer 3 → Fully Connedted Layer 4
* Output : size 10, each element represents propability for that class(label)

## Improvements
* Drop out: Added Dropout layer between two fully connected layers to prevent overfitting. (두 개의 완전연결층 사이에 Dropout 함수를 사용해 Drop Out 기법을 적용했다. 과적합을 방지하기 위함이다.)

```py
self.dropout = nn.Dropout(p = 0.5)
```

* Batch Normalization(배치 정규화): Added batch normalization layer between convolution layer and activation function. (합성곱층과 활성화 함수 사이에 배치 정규화를 진행했다. 보다 고른 값을 활성화함수로 넘길 수 있다.)

```py
self.bn0 = nn.BatchNorm2d(6)
self.bn1 = nn.BatchNorm2d(16)
self.bn2 = nn.BatchNorm2d(120)
```

* Activation Function(활성화 함수): Replaced `tanh` function with `LeakyReLU`. (기존에 사용했던 tanh 함수 대신 LeakyReLU 함수를 사용했다.)

```py
self.leakRelu = nn.LeakyReLU(0.1)
```

* Initialize weights(가중치 초기화): Initialized weights of convolution layers and fully conned layers using Xavier Uniform function. (합성곱층과 완전연결층의 가중치를 xavier 방법으로 초기화했다.)

```py
torch.nn.init.xavier_uniform_(self.conv0.weight)
torch.nn.init.xavier_uniform_(self.conv1.weight)
torch.nn.init.xavier_uniform_(self.conv2.weight)
torch.nn.init.xavier_uniform_(self.fc3.weight)
torch.nn.init.xavier_uniform_(self.fc4.weight)
```

- - -

# Change in Layers and Transformation
## Problem
기존에 연습에 사용한 MNIST 데이터는 32 x 32 사이즈였다. 그러나 FashionMNIST는 크기가 28 x 28이기 때문에 `Transform` 역시 같은 사이즈로 해야 할 것 같다고 생각했다.

단순히 `Transform` 크기만 조정하면 오류가 발생한다.

```py
        def get_data(name="MNIST"):
            my_transform = transforms.Compose([
                transforms.Resize([28, 28]),        # 이 부분 수정
                transforms.ToTensor(), 
                transforms.Normalize((0.5,), (1.0,))
            ])

            #### MNIST 부분 생략 ####

            elif name=="FashionMNIST":
                download_root = "./fasion_mnist_dataset"

                train_dataset = FashionMNIST(
                    root=download_root,
                    transform=my_transform,
                    train=True,
                    download=args.download)
                eval_dataset = FashionMNIST(
                    root=download_root,
                    transform=my_transform,
                    train=False,
                    download=args.download
                )
                test_dataset = FashionMNIST(
                    root=download_root,
                    transform=my_transform,
                    train=False,
                    download=args.download
                )
```
       
```
RuntimeError: Calculated padded input size per channel: (4 x 4). Kernel size: (5 x 5). Kernel size can't be greater than actual input size
```

이는 모델의 Layer을 고려하지 않아 발생한 문제이다. 에러 메시지를 보면 한 채널의 입력 크기는 (4x4)인데 커널 크기가 그보다 더 큰 (5x5)라 하고 있다.

## Problem Solving
따라서 레이어의 변경을 추가했다. 커널 사이즈를 5에서 3으로 줄이고 마지막 합성곱층 직후 풀링층 하나를 더 추가했다.(`2 --[ pool2 ] --> 1`) 그러면 아래와 같은 흐름을 보이고, 에러가 발생하지 않는다.

```
28 --[ conv0 ]--> 26 
   --[ pool0 ]--> 13 
   --[ conv1 ]--> 10 
   --[ pool1 ]--> 5 
   --[ conv2 ]--> 2
   --[ pool2 ]--> 1
```

또는 패딩을 2로 주어도 에러가 나지 않는다.

## Sevaral Cases
![](readme_imgs/losses_assign.png)

> 지금은 grayscale의 작은 이미지이기 때문에 하이퍼 파라미터를 변경해도 큰 성능 변동이 없을 수 있다.

## Predictions

![](readme_imgs/pred.png)

- - -

# Further Study
물론 32x32으로 Transform을 해도 충분히 실행은 된다. 이에 관한 팀원과 강사님의 코멘트는 몇 가지 Insight를 주었다.

## 1. Input Size
💡<u>**입력 크기의 변화는 정확도나 손실의 차이에는 영향을 크게 미치지 않는다. 대신 연산 속도나 파라미터 크기에 영향을 준다**</u>

이를 확인하기 위해 같은 조건에서 모델의 파라미터를 확인해보았다. `torchsummary` 라이브러리를 활용했으며, 결과는 아래와 같다.

일단, 28 x 28 크기로 변환해 모델에 입력했을 때의 결과이다. 파라미터가 2만 9천개 정도 된다.

![](readme_imgs/28.jpg)

그럼 32 x 32 모델을 유지했을 때는 어떻게 되었을까. 결과는 파라미터 개수가 거의 6만 2천개에 가까웠다.

![](readme_imgs/32.jpg)

모델의 성능에 영향을 미치는 파라미터 개수가 급격히 줄어든 것을 볼 수 있었다. 따라서 작동 여부를 떠나 입력 크기를 줄이면 성능 저하가 일어날 위험이 있는 것이다. 실제로 Loss와 정확도를 비교해보면 성능 차가 있었는데, 아래 Loss 그래프에서 확연히 드러난다.

![](readme_imgs/loss_compare.png)

scale이 다른 것은 눈을 감아주시길 바란다. 😭 ~하도 시도를 많이 하다보니...~ 좌측은 입력 사이즈가 작았을 때(28x28) 우측은 컸을 때이다. 좌측에서 손실이 1.4 ~ 1.6 에서 떨어지지 않았다. 그러나 우측에서는 거의 0에 가까울 때도 많을 만큼 손실이 많이 줄고 있다.

사실 LeNet-5의 기본 입력 사이즈는 32 X 32이다. 입력 사이즈를 변경하면 weight params 수가 크게 줄고, 이는 곧 모델의 capacity가 작아지고 더 가벼운 모델이 된다. Lenet-5 자체가 가벼운 모델이므로 더 줄일 필요는 적다. 또한 대체적으로 입력 사이즈가 클 수록 성능이 향상될 수 있다.

## 2. Softmax Func.
Lenet-5의 `forward()` 부분에서 softmax 함수를 지우면 loss가 더 수렴한다. 계산에 사용되는 CrossEntrophyLoss 내에 포함되어 있어 중복 계산이 줄기 때문이다.

## 3. Image Color
Matplotlib 라이브러리를 사용해 이미지를 출력하면 grayscale인 입력 이미지가 출력에서는 푸르스름하게 나온다. PIL을 이용하면 원래대로 Grayscale이 된다.

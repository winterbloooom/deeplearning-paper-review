import torch 
import torch.nn as nn

class LeNet5(nn.Module):
    # torch.nn.Module 클래스: 
    #   모든 신경망 모듈을 위한 base class. 모델을 만들려면 이 클래스를 subclass 해야 함
    #   Module 클래스는 다른 Module 들을 포함할 수 있음
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train=False):
        super().__init__()

        self.batch = batch
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train

        ### convolution & pooling layer
        # ※ convolution output = {(W - K + 2P) / S} + 1
        self.conv0 = nn.Conv2d(self.in_channel, 6, kernel_size=5, stride=1, padding=0)
            # torch.nn.Conv2d 클래스: 입력 신호에 대해 2D Convolution을 적용함
            #   필수 파라미터: input_channel, output_channel
            #   보통 입력 크기는 (B, C_in, H, W), 출력 크기는 (B, C_out, H_out, W_out)
            #   자료형으로 TensorFloat32를 사용
        self.pool0 = nn.AvgPool2d(2, stride=2)
            # torch.nn.AvgPool2d 클래스: 입력 신호에 대해 2D average pooling을 적용함
            #   필수 파라미터: kernel_size
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(2, stride=2)

        ### fully-connected layer
        self.fc3 = nn.Linear(120, 84)
            # torch.nn.Linear 클래스: 입력 데이터에 대해 선형 변환을 적용. y = x * A.t + b
            #   필수 파라미터: in_features(각 input sample의 사이즈), out_features
        self.fc4 = nn.Linear(84, self.n_classes)

        ### activation function
        self.leakRelu = nn.LeakyReLU(0.1)
            # torch.nn.LeakyReLU 클래스: LeakyReLU를 수행할 클래스. 
            #   인자 negative_slope는 기본값 1e-2로 설정되어 있음

        ### batch normalization
        self.bn0 = nn.BatchNorm2d(6)    # conv0의 output 채널이 6이기 때문에 인자로 6 전달
            # torch.nn.BatchNorm2d 클래스: 배치 정규화 수행. gamma, beta 파라미터 벡터를 갱신할 때 쓰기 위함
            #   파라미터 num_features: [batch, channel, height, width] 형상에서 channel을 의미
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(120)

        ### weight initialization(가중치 초기화)
        torch.nn.init.xavier_uniform_(self.conv0.weight)
            # torch.nn.init.xavier_uniform_() 함수: 가중치를 초기화하는 방법 중 하나
            # 가중치(weight)는 torch.nn.parameter.Parameter 형으로 torch.nn.modules.conv.Conv2d 클래스 내부에 있음(?)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        ### drop out
        self.dropout = nn.Dropout(p = 0.5)
            # torch.nn.Dropout 클래스: 
            #   파라미터 p: 어떤 weight를 0으로 만들지 그 비율을 정함. 기본은 0.5


    def forward(self, x):
        # nn.Module.forward() 메서드: 매 call마다 수행될 내용. subclass에서 재정의되어야 함. 재정의한 부분이 하단
        #   Model 객체를 데이터와 함께 호출하면 자동으로 실행됨. 따라서 my_model = LeNet(input)으로 선언/호출 해도 자동으로 forward 수행됨
        # ※ x의 shape: [B, C, H, W]

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.leakRelu(x)    #x = torch.tanh(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakRelu(x)    #x = torch.tanh(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakRelu(x)    #x = torch.tanh(x)
        #x = self.pool2(x)      # 이미지 크기 변경 시 풀링 층 추가
        
        x = torch.flatten(x, start_dim=1)
            # 4차원을 2차원으로 바꿈 ([b, c, h, w] -> [B, C*H*W])

        x = self.fc3(x)
        x = self.dropout(x)
        x = self.leakRelu(x)    #x = torch.tanh(x)
        x = self.fc4(x)

        x = x.view(self.batch, -1)
            # Tensor.view() 메서드: 인자로 주어진 tensor의 shape를 변경해 새 tensor로 리턴
            #   파라미터 shape에 -1 입력 시, 다른 dimension으로부터 값을 자동으로 추정함

        # x = nn.functional.softmax(x, dim=1)
            # torch.nn.functional.softmax() 함수: [0, 1]의 범위를 갖도록 softmax 함수를 적용
            # 추후 MNISTLoss의 CrossEntropyLoss에 소프트맥스가 포함되어 있어 중복 계산이 될 수도 있음
            #   따라서 여기서 주석처리로 제거하여 더 수렴할 수 있도록 변경함

        if self.is_train is False:
            x = torch.argmax(x, dim=1)
                # torch.argmax() 함수: 입력 tensor에 대해 가장 큰 값을 리턴함

        return x
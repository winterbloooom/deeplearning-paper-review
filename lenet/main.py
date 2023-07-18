import torch
import torch.nn as nn
    # 그래프의 기본적 building blocks를 제공
from torchvision.datasets import MNIST, FashionMNIST
    # torchvision: 유명한 데이터셋, 모델 아키텍처, 이미지 변환 등을 제공
    # torchvision.datasets: 빌트인 데이터셋을 제공하는 모듈
import torchvision.transforms as transforms
    # 이미지 transform을 제공하는 모듈
from torch.utils.data.dataloader import DataLoader
    # torch.utils.data.DataLoader: 데이터 로딩 유틸리티 클래스
import torch.optim as optim
    # optimization 알고리즘들을 제공하는 패키지

import pytorch_model_summary
    # 모델의 output shape, 파라미터 수를 확인하기 위해 불러옴
import matplotlib.pyplot as plt
    # 학습 중의 loss를 시각화하기 위해 불러옴
    
import argparse
    # sys.argv를 어떻게 parsing할 지 파악하는 등의 역할을 함
import sys, os

from lenet5 import LeNet5
from loss import *
from tools import *

### command-line에서 사용자가 입력한 arguments 들을 parsing
### -> mode, 데이터셋 download 여부, output 파일 저장 위치, checkpoint 파일 지정
def parse_args():
    parser = argparse.ArgumentParser(description="FashionMNIST")
        # ArgumentParser 객체: command-line을 파이썬 데이터형으로 parsing할 수 있는 객체

    parser.add_argument('--mode',
                        dest = 'mode',
                        help = 'train / eval / test',
                        default = None,
                        type = str)
        
    parser.add_argument('--download',
                        dest = 'download',
                        help = 'download dataset',
                        default = False,
                        type = bool)
    parser.add_argument('--output_dir',
                        dest = 'output_dir',
                        help = 'output directory',
                        default = './output',
                        type = str)
    parser.add_argument('--checkpoint',
                        dest = 'checkpoint',
                        help = 'checkpoint trained model',
                        default = None,
                        type = str)
    parser.add_argument('--dataset',
                        dest = 'dataset',
                        help = 'dataset to train model',
                        default = None,
                        type = str) # MNIST, FashionMNIST
        # add_argument() 메서드: ArgumentParser이 command-line의 문자을 객체로 어떻게 변환시킬지 지정

    ### 사용자가 args를 지정하지 않았다면 올바른 사용법 출력 후 종료
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()  
        # ArgumentParser.parse_args() 메서드: 인자 문자열을 객체로 변환 & namespace의 attribute로 설정
        # 예: Namespace(checkpoint=None, download=True, mode='train', output_dir='./output') 식으로 나옴
        # args는 <class 'argparse.Namespace'> 타입을 가짐

    return args

def get_data(name="MNIST"):
    my_transform = transforms.Compose([
        transforms.Resize([32, 32]), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])
        # torchvision.transforms.Compose 클래스: 파라미터로써 compose를 할 Transform 객체의 리스트인 transforms를 받음
        # torchvision.transforms.Resize 클래스: 입력 이미지를 특정 사이즈로 변환함
        # torchvision.transforms.ToTensor 클래스: [0, 255] 범위의 (H X W X C)의 PIL 이미지나 numpy.ndarray를 [0.0, 1.0] 범위의 (C X H X W) float tensor로 변경함
        # torchvision.transforms.Normalize 클래스: 
        #   tensor 이미지를 mean & standard deviation으로 정규화시킴
        #   파라미터 mean = 각 채널의 mean의 sequence / 파라미터 std = 각 채널의 standard deviation의 sequence
    
    if name=="MNIST":
        download_root = "./mnist_dataset"

        train_dataset = MNIST(
            root=download_root,
            transform=my_transform,
            train=True,
            download=args.download)
        eval_dataset = MNIST(
            root=download_root,
            transform=my_transform,
            train=False,
            download=args.download
        )
        test_dataset = MNIST(
            root=download_root,
            transform=my_transform,
            train=False,
            download=args.download
        )
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
            # torchvision.datasets.FashionMNIST 클래스 파라미터
            #   root: 데이터들의 root 디렉터리
            #   train: train된 파일에서 불러와 데이터셋을 만들지 여부
            #   download: True로 설정 시, 인터넷에서 데이터를 다운받아 root dir에 넣음
            #   transform: PIL 이미지를 받아 변형된 형태로 리턴하는 함수나 transform

    return train_dataset, eval_dataset, test_dataset

def make_dataloader(train_dataset, eval_dataset, test_dataset):
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)
        # torch.utils.data.DataLoader 클래스: 데이터셋과 sampler를 결합하고, 주어진 데이터셋에 대해 iterable을 제공
        #   사용 파라미터:
        #       batch_size: 한 에폭마다 들어갈 배치의 수. 기본은 1
        #       num_workers: 데이터 로딩에 쓸 subprocesses의 개수. 0이면 메인 프로세스에서 불러옴을 의미. 본인의 CPU, GPU 상황에 따라 주는 값을 달리하기
        #       drop_last : True로 설정 시, batch 사이즈를 맞추고 남은 데이터들을 drop 시킴. 디폴트는 False로, 남는 데이터들은 조금 작은 배치 크기를 형성함
        #       pin_memory: True로 설정 시, 데이터 로더는 Tensor를 CUDA pinned momory로 복사함.
        #       shuffle: False가 default이며, True로 설정 시 한 에폭이 끝날 때마다 데이터들의 순서를 섞음
        #   자료형 구성:
        #       [0]: tensor 형의 img가 batch_size개 있음
        #       [1]: tensor 형의 정답 라벨이 batch_size개 있음
        # eval, test는 평가 용이기 때문에 batch_size를 1로. drop_last, shuffle이 굳이 필요 없음
    
    return train_loader, eval_loader, test_loader

def train_model(_model, device, train_loader, in_width, in_height, epoch=15):
    model = _model(batch=8, n_classes=10, in_channel=1, in_width=in_width, in_height=in_height, is_train=True)
    model.to(device)
        # Module.to() 메서드: 파라미터와 버퍼들을 옮김(또는 cast). device 파라미터에는 torch.device 객체가 들어감
    model.train()
        # Module.train() 메서드: 모듈을 training 모드로 설정함. 

    ### optimizer(최적화): 최적(가장 좋은 성능을 보이는)의 파라미터를 찾는 방식 설정
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        # torch.optim 패키지를 사용하기 위해선 현재 상태(state)를 저장 & 파라미터들을 업데이트(계산된 기울기에 따라)할 optimizer 객체를 선언해야 함
        # Module.parameters() 메서드: 모듈의 파라미터들에 대한 iterator를 리턴함. 보통 optimizer에게 전달됨.
        #   nn.Parameter: Tensor 형을 가짐. Module에 속성으로 할당 시 자동으로 매개변수로 등록됨
        # torch.optim.SGD 클래스: stochastic gradient descent를 적용함
        #   lr(=learning rate), momentum(default: 0), weight_decay(regularization 항 추가 위한 파라미터)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    ### learning rate scheduler: 미리 학습 일정을 정해두고 그에 따라 학습률을 조정함 -> 상황에 맞게 학습률을 가변적으로 적당히 변경
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # torch.optim.lr_scheduler: 에폭의 수에 따라 학습률을 조정하도록 함. 최적화 업데이트 이후 수행되어야 함
        # torch.optim.lr_scheduler.StepLR 클래스:
        #   지정한 step_size 에폭 단위로 학습률에 gamma를 곱해 학습률을 감소시킴
        #   step_size: 학습률 decay의 주기 / gamma: 기본은 0.1, 학습률 decay의 mulitplicative 인자

    ### criterion: 손실함수 클래스 불러오기
    criterion = MNISTloss(device = device)

    ### 학습에 사용될 각종 수 설정
    epoch = epoch
    iter = 0

    ### loss 그래프 확인을 위한 변수
    losses = []

    ### 각 epoch마다 학습 진행
    for e in range(epoch):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            ### DataLoader에서 불러온 데이터셋을 가져온 뒤 디바이스에 올리기
            img = batch[0]  # shape 예: torch.Size([8, 1, 32, 32]) -> [배치, 채널, 세로, 가로]
            gt = batch[1]   # (예) tensor([8, 4, 0, 9, 9, 2, 9, 2])
            img = img.to(device)
            gt = gt.to(device)

            ### 모델에서의 output을 받아옴
            out = model(img)
                # torch.Size([8, 10]) ==> [배치, 클래스 수] 형태로 각 배치에서 클래스별 결과(내놓은 답)을 출력
            
            ### output과 정답을 비교해서 손실함수를 사용해 손실값을 구함
            loss_val = criterion(out, gt)

            ### 손실값을 사용해 오차역전파를 사용해 파라미터를 개선시킴
            loss_val.backward()
                # Module 클래스에서 forward 함수만 정의하면,
                # backward 함수는 'torch.autograd' 기능(신경망 학습을 지원하는 PyTorch의 자동 미분 엔진)을 통해 자동으로 정의됨
            optimizer.step()
                # torch.optim.SGD.setp() 메서드: single 최적화 setp을 수행 -> 가중치 업데이트
            optimizer.zero_grad()
                # 모든 매개변수의 gradient 버퍼를 0으로 만듦
                #  ==> 기존의 계산된 기울기 값을 누적시키지 않을 때, 기존에 계산된 기울기를 0으로 만듦

            ### 전체 손실값을 누적시킴
            total_loss += loss_val.item()
                # torch.tensor.item(): tensor의 값을 파이썬 표준 number로 리턴함. tensor에 하나의 elem만 있어야 함
            
            ### 일정 iter마다 loss를 출력해봄
            if iter % 100 == 0:
                print("{} epoch {} iter loss: {}".format(e, iter, loss_val.item()))
                losses.append(loss_val.item())

            iter += 1
        
        ### 한 에폭이 끝남 -> 평균 손실을 구하며 학습률 갱신, 텐서 저장
        mean_loss = total_loss / i  # 평균 손실값을 구함, i = train_loader size:  7499
        scheduler.step()    # 학습률을 갱신함
        print("-> {} epoch mean loss: {}".format(e, mean_loss))
        torch.save(model.state_dict(), args.output_dir + "/model_epoch"+str(e)+".pt")
            # torch.save() 함수: 파일로 객체를 저장함
            #   obj: 저장할 객체 / f: 파일 이름(관습적으로 *.pt 파일을 사용해 tensor를 저장함)
            # Module.state_dict() 메서드: 모듈의 모든 state를 담고 있는 딕셔너리를 리턴함
            #   파라미터들과 버퍼들을 포함함. key는 파라미터와 버퍼의 이름
            #   (ex) module.state_dict().keys()의 출력 = ['bias', 'weight']
    
    print(pytorch_model_summary.summary(model, torch.zeros(8, 1, 32, 32), max_depth=None, show_parent_layers=True, show_input=True))
    plt.plot(np.arange(iter // 100), losses, 'b', label='loss')
    plt.ylim([0, 2.5])
    plt.show()

    print("===== END Train =====")


def eval_model(_model, device, eval_loader, in_width, in_height):
    model = _model(batch=1, n_classes=10, in_channel=1, in_width=in_width, in_height=in_height)
    checkpoint = torch.load(args.checkpoint)    # trained model을 load함
    model.load_state_dict(checkpoint)
        # Module.load_state_dict() 메서드: state_dict로부터 해당 모듈과 그 자식들로 파라미터와 버퍼들을 복사해옴
    model.to(device)
    model.eval()
        # Module.eval() 메서드: 모듈을 evaluation mode로 설정함

    ### 평가에 쓰일 수 정의
    accuracy = 0    # 정확도
    num_eval = 0    # 평가 횟수

    ### load한 데이터셋마다 학습 진행
    for batch in eval_loader:
        img = batch[0]
        gt = batch[1]
        img = img.to(device)

        out = model(img)
        out = out.cpu()
            # Module.cpu() 메서드: 모든 모델의 파라미터 & 버퍼를 CPU로 옮김

        if out == gt:
            accuracy += 1    # 모델이 낸 답이 정답과 같으면 정확도 +1

        num_eval += 1
    
    print("Evaluation scroe: {} / {} -> {} %".format(accuracy, num_eval, accuracy/num_eval))
        # 총 몇 개의 정답을 맞추었는가? (예) Evaluation scroe: 8682 / 10000

def test_model(_model, device, test_loader):
    model = _model(batch=1, n_classes=10, in_channel=1, in_width=1, in_height=1)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    cnt = 0
    for batch in test_loader:
        if cnt >= 5:
            break

        img = batch[0]
        gt = batch[1]
        img = img.to(device)

        out = model(img)
        out = out.cpu()

        ### 결과를 이미지로 출력함
        show_img(img.cpu().numpy(), str(out.item()))
            # Tensor.numpy(): tensor를 ndarray로 변경함
            # 이미지 위에 모델이 낸 답을 씀
        print("ans: {}".format(str(gt.item())))

        cnt += 1

def main():
    ### output_dir로 설정한 디렉터리가 존재하는 지 확인 -> 존재하지 않는다면 만들기
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    ### 어떤 device에 올릴지 지정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ### FashionMNIST dataset을 불러옴
    train_dataset, eval_dataset, test_dataset = get_data(name=args.dataset)

    ### DataLoader를 만듦
    train_loader, eval_loader, test_loader = make_dataloader(train_dataset, eval_dataset, test_dataset)

    ### LeNet5 모델을 불러옮
    _model = LeNet5

    ### 모델 학습
    if args.dataset == "MNIST":
        in_width, in_height = 32, 32
    elif args.dataset == "FashionMNIST":
        in_width, in_height = 28, 28

    if args.mode == "train":
        train_model(_model, device, train_loader, in_width, in_height, epoch=5)
    elif args.mode == "eval":
        eval_model(_model, device, eval_loader, in_width, in_height)
    elif args.mode == "test":
        test_model(_model, device, test_loader)

if __name__=="__main__":
    args = parse_args()
    main()
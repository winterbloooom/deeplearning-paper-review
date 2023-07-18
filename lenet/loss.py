import torch
import torch.nn as nn

class MNISTloss(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        super(MNISTloss, self).__init__()
            # super(자식 클래스, self)로 부모 클래스의 메서드 호출. super()와 동일. 현재 클래스가 어떤 클래스인지 명확히 표시
        self.loss = nn.CrossEntropyLoss().to(device)
            # torch.nn.CrossEntropyLoss 클래스: input과 target 간의 cross entropy loss를 계산하는 criterion. 분류 문제에 많이 사용됨.

    def forward(self, out, gt):
        # nn.Module.forward() 메서드: 매 call마다 수행될 내용. subclass에서 재정의되어야 함. 재정의한 부분이 하단
        #   Model 객체를 데이터와 함께 호출하면 자동으로 실행됨. 따라서 my_model = LeNet(input)으로 선언/호출 해도 자동으로 forward 수행됨
        loss_val = self.loss(out, gt)
            # loss(input, target) 식으로 사용
            # 클래스(분류 목록)에 대한 확률로 target이 주어진다면, shape는 input과 같아야 하고 각 값은 [0, 1] 범위의 값이어야 함
            # CrossEntropyLoss() 선언 시 reduction이 none으로 설정되었다면 target과 같은 shape를, 그렇지 않다면 sclar 반환
        
        return loss_val
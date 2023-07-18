import torch.nn as nn


class Generator(nn.Module):
    """생성자(Generator) model"""

    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 100

        def block(input_dim, output_dim, normalize=True):
            """하나의 블록(선형+배치정규화+활성화함수)을 만듦"""
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))  # 배치 정규화 추가
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # 활성화함수
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )
        # 최종 출력은 28*28의 이미지
        # 파이썬에서 "*" 기호는 unpack을 의미. 딕셔너리를 풀려면 "**"

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    """판별자(Discriminator) model"""

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # 마지막 시그모이드 함수로 판별 결과 내보냄

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)

        return output

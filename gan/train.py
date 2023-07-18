import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image


def get_lr(optimizer):
    """최적화 함수의 learning rate를 반환함"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_model(Generator, Discriminator, device, train_loader, n_epochs, output_dir, torch_writer):
    """생성자와 판별자를 훈련함

    Args:
        Generator (Generator) : 생성자 모델
        Discriminator (Discriminator) : 판별자 모델
        device (torch.device) : GPU 혹은 CPU
        train_loader (torch.utils.data.DataLoader) : 훈련 데이터의 데이터 로더
        n_epochs (int) : 총 에폭 수
        output_dir (str) : 결과물을 저장할 폴더
        torch_writer (SummaryWriter) : TensorBoard의 출력을 저장
    """
    print("\n" + "=" * 50 + " Train Model " + "=" * 50)

    # 생성자
    G = Generator.to(device)
    G.train()
    print(G)

    # 판별자
    D = Discriminator.to(device)
    D.train()
    print(D)

    # 설정값
    lr = 0.0002  # 학습률
    sample_interval = 200  # 결과 출력 단위
    latent_dim = 100
    start_time = time.time()  # 학습 시작 시간
    iter = 0  # 반복 횟수

    # 최적화함수
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # 손실함수
    adversarial_loss = nn.BCELoss()
    adversarial_loss.to(device)

    # 학습 진행
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성
            real = torch.Tensor(imgs.size(0), 1).fill_(1.0)  # 진짜(real): 1
            fake = torch.Tensor(imgs.size(0), 1).fill_(0.0)  # 가짜(fake): 0

            real_imgs = imgs.to(device)  # 실제 이미지(GT)

            # 생성자 학습===================================================
            optimizer_G.zero_grad()  # 기울기 초기화
            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).to(
                device
            )  # 랜덤 노이즈(G 입력) 샘플링
            generated_imgs = G(z)  # 이미지 생성
            g_loss = adversarial_loss(D(generated_imgs), real)  # 손실 (생성v실제)

            # 생성자 업데이트================================================
            g_loss.backward()  # 역전파
            optimizer_G.step()  # 최적화

            # 판별자 학습===================================================
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(D(real_imgs), real)  # 손실(실제)
            fake_loss = adversarial_loss(D(generated_imgs.detach()), fake)  # 손실(생성vs가짜)
            d_loss = (real_loss + fake_loss) / 2

            # 판별자 업데이트================================================
            d_loss.backward()  # 역전파
            optimizer_D.step()  # 최적화

            # 일정 간격마다 결과 출력
            if iter % sample_interval == 0:
                save_image(
                    generated_imgs.data[:25],
                    output_dir + f"/img/{iter}.png",
                    nrow=5,
                    normalize=True,
                )  # 25개 결과물을 이미지로 저장
                print(
                    "{:>2} / {:>2} epoch ({:>3.0f} %) {:>3} iter \t G loss: {:>.8f} \t D loss: {:>.8f}".format(
                        epoch,
                        n_epochs,
                        100.0 * i / len(train_loader),
                        iter,
                        g_loss.item(),
                        d_loss.item(),
                    )
                )
                torch_writer.add_scalar("G lr", get_lr(optimizer_G), iter)
                torch_writer.add_scalar("G Loss", g_loss.item(), iter)
                torch_writer.add_scalar("G lr", get_lr(optimizer_D), iter)
                torch_writer.add_scalar("D Loss", d_loss.item(), iter)

            iter += 1

        print(
            f"-> [Epoch {epoch:3d} / {n_epochs:3d}] D loss: {d_loss.item():.6f} | G loss: {g_loss.item():.6f} (Elapsed time: {time.time() - start_time:8.2f} s)"
        )

    print("=" * 50 + " Train End " + "=" * 50)

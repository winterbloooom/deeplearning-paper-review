import argparse
import os
import sys

import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets

from models import *
from train import train_model


def parse_args():
    """프로그램 시작 시 arguments를 parsing함"""
    parser = argparse.ArgumentParser(description="GAN-with-pytorch")
    parser.add_argument(
        "--download", dest="download", help="download dataset", default=False, type=bool
    )
    parser.add_argument(
        "--dataset_dir",
        dest="dataset_dir",
        help="dataset download directory",
        default="./dataset",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="output directory",
        default="./output",
        type=str,
    )
    parser.add_argument(
        "--log_num",
        dest="log_num",
        help="trial number",
        default="1",
        type=str,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    return args


def get_data(dataset_dir, download):
    """get dataset"""
    transforms_train = transforms.Compose(
        [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    train_dataset = datasets.MNIST(
        root=dataset_dir, train=True, download=download, transform=transforms_train
    )

    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    print("train_dataset: " + str(train_dataset.data.shape))
    return dataloader


def main():
    print(f"=" * 50 + " Start (Log Num: {args.log_num})" + "=" * 50)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    print("output directory: " + args.output_dir)

    if not os.path.exists(args.dataset_dir):
        os.mkdir(args.dataset_dir)
    print("dataset directory: " + args.dataset_dir)

    # 기기 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device: " + device.type)

    # 데이터 로드
    print("\n" + "=" * 50 + " Data Load " + "=" * 50)
    train_loader = get_data(dataset_dir=args.dataset_dir, download=args.download)

    # 모델 선언
    generator = Generator()
    discriminator = Discriminator()

    torch_writer = SummaryWriter(logdir="output/summary/" + args.log_num)
    train_model(
        Generator=generator,
        Discriminator=discriminator,
        device=device,
        train_loader=train_loader,
        n_epochs=200,
        output_dir=args.output_dir + "/" + args.log_num,
        torch_writer=torch_writer,
    )
    torch_writer.close()


if __name__ == "__main__":
    args = parse_args()
    main()

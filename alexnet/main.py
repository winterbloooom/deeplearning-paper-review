import argparse
import os
import sys
from test.test import *

import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets

from model.alexnet import AlexNet
from train.train import *
from utils.tools import *


# parsing commandline args
def parse_args():
    parser = argparse.ArgumentParser(description="AlexNet-with-pytorch")
    parser.add_argument(
        "--mode", dest="mode", help="train / test", default="test", type=str
    )
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
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint trained model",
        default=None,
        type=str,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    return args


# dataset download
def get_data(dataset_dir, download):
    img_transform = transforms.Compose(
        [
            transforms.Resize(227),
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=[227,]),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    train_dataset = datasets.STL10(
        root=dataset_dir, split="train", download=download, transform=img_transform
    )
    print("train_dataset: " + str(train_dataset.data.shape))  # (5000, 3, 96, 96)

    test_dataset = datasets.STL10(
        root=dataset_dir, split="test", download=download, transform=img_transform
    )
    print("train_dataset: " + str(test_dataset.data.shape))  # (8000, 3, 96, 96)

    return train_dataset, test_dataset


# data loader
def make_dataloader(train_dataset, test_dataset):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(dataset=test_dataset)
    return train_loader, test_loader


def main():
    print("=" * 50 + " Start " + "=" * 50)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    print("output directory: " + args.output_dir)

    if not os.path.exists(args.dataset_dir):
        os.mkdir(args.dataset_dir)
    print("dataset directory: " + args.dataset_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device: " + device.type)

    print("\n" + "=" * 50 + " Data Load " + "=" * 50)
    train_dataset, test_dataset = get_data(
        dataset_dir=args.dataset_dir, download=args.download
    )
    train_loader, test_loader = make_dataloader(
        train_dataset=train_dataset, test_dataset=test_dataset
    )

    model = AlexNet
    
    if args.mode == "train":
        torch_writer = SummaryWriter(logdir="output/exp_SGD_1")
        train_model(
            _model=model,
            device=device,
            train_loader=train_loader,
            batch=64,
            n_classes=10,
            in_channel=3,
            in_width=227,
            in_height=227,
            _epoch=5,
            output_dir=args.output_dir,
            torch_writer=torch_writer,
        )
        torch_writer.close()

    elif args.mode == "test":
        test_model(
            _model=model,
            device=device,
            test_loader=test_loader,
            batch=1,
            n_classes=10,
            in_channel=3,
            in_width=227,
            in_height=227,
            _checkpoint=args.checkpoint,
        )


if __name__ == "__main__":
    args = parse_args()
    main()

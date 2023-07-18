import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train=False):
        super().__init__()

        self.batch = batch
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.is_train = is_train

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channel, out_channels=96, kernel_size=11, stride=4, padding=0,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=(6 * 6 * 256), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=self.n_classes)

        self.relu = nn.ReLU(inplace=True)

        self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

        self.dropout = nn.Dropout(p=0.5)

        torch.nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv4.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.conv5.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc3.weight, mean=0, std=0.01)

        torch.nn.init.constant_(self.conv1.bias, 0)
        torch.nn.init.constant_(self.conv2.bias, 1)
        torch.nn.init.constant_(self.conv3.bias, 0)
        torch.nn.init.constant_(self.conv4.bias, 1)
        torch.nn.init.constant_(self.conv5.bias, 1)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.view(-1, 256 * 6 * 6)
        # print(x.shape)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        # x = nn.functional.softmax(x, dim=1)

        if self.is_train is False:
            # if mode is 'test', only one answer is needed, not all probabilities for all classes
            x = torch.argmax(x, dim=1)

        # print("output shape: " + str(x.shape))
        return x

import torch
import torch.nn as nn


class AudioNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool2d_2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(267488, 50)
        self.fc2 = nn.Linear(50, 25)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool2d_1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.max_pool2d_2(x)
        x = self.relu2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)

        return x


if __name__ == "__main__":

    network = AudioNetwork()
    input = torch.ones(16, 2, 64, 2584, requires_grad=True)
    output = network(input)

    print("done")

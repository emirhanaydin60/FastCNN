import torch
import torch.nn as nn


class FastCNN(nn.Module):
    """Parametrizable CNN for MNIST.

    Parameters configurable at construction time:
    - in_channels (default 1)
    - conv_filters: tuple/list (f1, f2)
    - kernel_size: conv kernel (default 3)
    - pool_kernel, pool_stride: maxpool parameters
    - fc1_size: number of neurons in first FC
    - dropout_p: dropout probability
    - num_classes: output classes

    The model returns logits (no final softmax) so it's compatible with
    `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_filters=(16, 32),
        kernel_size: int = 3,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        fc1_size: int = 100,
        dropout_p: float = 0.5,
        num_classes: int = 10,
    ):
        super().__init__()
        f1, f2 = conv_filters
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(f2, fc1_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc1_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # quick shape sanity check
    model = FastCNN()
    x = torch.randn(4, 1, 28, 28)
    print(model(x).shape)

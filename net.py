import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

def custom_batch_norm(x, num_channels, bn_layer):
    x = x[:, :num_channels]
    running_mean = bn_layer.running_mean[:num_channels]
    running_var = bn_layer.running_var[:num_channels]
    weight = bn_layer.weight[:num_channels]
    bias = bn_layer.bias[:num_channels]

    x_normalized = (x - running_mean[None, :, None, None]) / torch.sqrt(running_var[None, :, None, None] + bn_layer.eps)
    return x_normalized * weight[None, :, None, None] + bias[None, :, None, None]



# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Custom Convolutional Neural Network with channel_multiplier
class CustomNet(nn.Module):
    def __init__(self, channel_multiplier):
        super(CustomNet, self).__init__()
        self.channel_multiplier = channel_multiplier
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        num_channels = int(self.channel_multiplier.item() * self.conv1.out_channels)
        weight = self.conv1.weight[:num_channels]
        bias = self.conv1.bias[:num_channels]

        x = F.conv2d(x, weight, bias, self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups)
        x = custom_batch_norm(x, num_channels, self.bn1)
        x = self.relu1(x)
        x = self.pool(x)
        input_size = num_channels * (x.size(1) // num_channels) * (x.size(2) * x.size(3))
        x = x.view(x.size(0), -1)

        # print(x.shape)

        # Update fc1 input size based on the current number of channels

        self.fc1 = nn.Linear(input_size, 128).to(device)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channel_multiplier = nn.Parameter(torch.tensor(1.0), requires_grad=True)
net = CustomNet(channel_multiplier).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params': list(net.parameters()) + [channel_multiplier]}], lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


def custom_loss(outputs, targets, channel_multiplier, alpha=2e-2):
    base_loss = criterion(outputs, targets)
    regularization_loss = alpha * torch.abs(channel_multiplier)
    total_loss = base_loss + regularization_loss
    return total_loss

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = custom_loss(outputs, labels, channel_multiplier)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        scheduler.step()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}, Channel Multiplier: {channel_multiplier.item()}')

print('Finished Training')

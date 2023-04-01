import torch.nn as nn
import torch

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, channel_multiplier):
        super(MBConvBlock, self).__init__()

        # Expansion phase
        self.expand = None
        if expand_ratio != 1:
            expanded_channels = int(round(in_channels * expand_ratio))
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True)
            )
            in_channels = expanded_channels

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, (kernel_size-1)//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )

        # Pointwise convolution
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Set requires_grad=True for the channel_multiplier tensor
        self.channel_multiplier = nn.Parameter(torch.tensor(channel_multiplier), requires_grad=True)

    def forward(self, x):
        identity = x

        if self.expand is not None:
            x = self.expand(x)

        x = self.depthwise(x)
        x = self.project(x * self.channel_multiplier)

        if x.shape == identity.shape:
            x += identity

        return x

# Create an instance of the MBConvBlock with channel multiplier 1.0
mbconv_block = MBConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, expand_ratio=6, channel_multiplier=1.0)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mbconv_block.parameters(), lr=0.1)

# Define the input and target tensors
input_tensor = torch.randn(1, 16, 224, 224)
target_tensor = torch.randn(1, 32, 224, 224)

# Perform channel search using gradient descent to update the channel multiplier
for epoch in range(100):
    # Compute the model output
    output_tensor = mbconv_block(input_tensor)

    # Compute the loss
    loss = criterion(output_tensor, target_tensor)

    # Compute the gradients
    optimizer.zero_grad()
    loss.backward()

    # Update the channel multiplier using gradient descent
    if mbconv_block.channel_multiplier.grad is not None:
        print("**************")
        with torch.no_grad():
            new_channel_multiplier = mbconv_block.channel_multiplier - mbconv_block.channel_multiplier.grad.detach() * 0.1
            new_channel_multiplier = torch.clamp(new_channel_multiplier, 0.1, 1.0)
            mbconv_block.channel_multiplier.copy_(new_channel_multiplier)
    else:
        print("____________________")

    # Update the optimizer
    optimizer.step()

    # Print the loss
    print("Epoch {}: loss={}, channel_multiplier={}".format(epoch, loss.item(), mbconv_block.channel_multiplier.item()))


"""
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = FBNetV2Example().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs, 3, 64, 64, 64)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        scheduler.step()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')

print('Finished Training')

"""

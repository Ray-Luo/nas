import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from blocks import FBNetV2BasicSearchBlock
from tqdm import tqdm



in_channels = 3
num_masks = 3

conv_kernel_configs2 = [
    [3, 1, 1, 0, 0],
    [3, 1, 1, 0, 1],
    [3, 1, 1, 1, 0],
    [3, 1, 1, 1, 1],
    [5, 1, 1, 0, 0],
    [5, 1, 1, 0, 1],
    [5, 1, 1, 1, 0],
    [5, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
]
conv_kernel_configs3 = [
    [3, 1, 1, 0, 0],
    [3, 1, 1, 0, 1],
    [3, 1, 1, 1, 0],
    [3, 1, 1, 1, 1],
    [5, 1, 1, 0, 0],
    [5, 1, 1, 0, 1],
    [5, 1, 1, 1, 0],
    [5, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
]
subsampling_factors = [1,2]
target_height = 32
target_width = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = FBNetV2BasicSearchBlock(32, max_out_channels=100, num_masks=3, conv_kernel_configs=conv_kernel_configs2, subsampling_factors=subsampling_factors, target_height=target_height, target_width=target_width)
        self.conv3 = FBNetV2BasicSearchBlock(100, max_out_channels=150, num_masks=3, conv_kernel_configs=conv_kernel_configs3, subsampling_factors=subsampling_factors, target_height=target_height, target_width=target_width)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(150 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x)))) # 100, 32, 32 --> 100, 16, 16
        x = self.pool(torch.relu(self.conv2(x))) # 150, 16, 16 --> 150, 16, 16
        x = self.pool(torch.relu(self.conv3(x))) # 150, 16, 16 --> 300, 16, 16
        x = x.view(-1, 150 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the transformations to apply to the input data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)


def train(net, dataloader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(dataloader, desc="Training")):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / (i + 1), 100 * correct / total

def evaluate(net, dataloader, criterion, device):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / (i + 1), 100 * correct / total


net = Net().to(device)

num_epochs = 1000
best_acc = 0
save_path = './logs_cifar10/best_cifar10.pth'
writer = SummaryWriter(log_dir='./logs_cifar10')

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}:")
    train_loss, train_acc = train(net, trainloader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(net, testloader, criterion, device)

    writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)
    writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(net.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

writer.close()
print(f"Training complete. Best accuracy: {best_acc:.2f}%")

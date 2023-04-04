import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from blocks import FBNetV2BasicSearchBlock
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets


a = torch.randn(1, 3, 256, 256)

in_channels = 3
max_out_channels = 256
num_masks = 3
# kernel_size, stride, expansion, use_se, use_hs
conv_kernel_configs1 = [
    [3, 1, 1, 0, 0],
]
conv_kernel_configs2 = [
    [3, 1, 1, 0, 1],
    [3, 1, 1, 1, 0],
    [3, 1, 1, 1, 1],
    [3, 1, 1, 0, 1],
    [5, 1, 1, 0, 1],
    [5, 1, 1, 1, 0],
    [5, 1, 1, 1, 1],
    [5, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
]
conv_kernel_configs3 = [
    [3, 1, 1, 0, 1],
    [3, 1, 1, 1, 0],
    [3, 1, 1, 1, 1],
    [3, 1, 1, 0, 1],
    [5, 1, 1, 0, 1],
    [5, 1, 1, 1, 0],
    [5, 1, 1, 1, 1],
    [5, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
]
subsampling_factors = [1,2,4,8]
target_height = 128
target_width = 128

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = FBNetV2BasicSearchBlock(3, max_out_channels=100, num_masks=3, conv_kernel_configs=conv_kernel_configs1, subsampling_factors=subsampling_factors, target_height=32, target_width=32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = FBNetV2BasicSearchBlock(32, max_out_channels=100, num_masks=3, conv_kernel_configs=conv_kernel_configs2, subsampling_factors=subsampling_factors, target_height=target_height, target_width=target_width)
        self.conv3 = FBNetV2BasicSearchBlock(100, max_out_channels=150, num_masks=3, conv_kernel_configs=conv_kernel_configs3, subsampling_factors=subsampling_factors, target_height=target_height, target_width=target_width)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(150 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 101)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x)))) # 100, 32, 32 --> 100, 16, 16
        # print(x.shape,"********")
        x = self.pool(torch.relu(self.conv2(x))) # 150, 16, 16 --> 150, 16, 16
        x = self.pool(torch.relu(self.conv3(x))) # 150, 16, 16 --> 300, 16, 16
        x = x.view(-1, 150 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.pool(self.relu1(out))

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.pool(self.relu2(out))

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.pool(self.relu3(out))

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu4(out)

        out = self.fc2(out)

        return out



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the transformations to apply to the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# load the CIFAR-10 dataset
full_dataset = datasets.Caltech101(root='./data', download=True, transform=transform)

train_size = int(0.8 * len(full_dataset))
eval_size = len(full_dataset) - train_size

# split the dataset into training and validation sets
torch.manual_seed(42)
train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

# create the data loaders
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
testloader = DataLoader(eval_dataset, batch_size=4)


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
save_path = './best_caltech.pth'
writer = SummaryWriter(log_dir='./logs_caltech')

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

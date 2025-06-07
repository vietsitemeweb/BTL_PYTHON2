# BTL Python 2 - Trần Minh Việt & Nguyễn Thiên Trung
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_loaders(batch_size=64, val_ratio=0.1):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    val_set.dataset.transform = transform_test

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128*4*4, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / total, 100 * correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    return running_loss / total, 100 * correct / total, targets, preds

def plot_learning_curve(epochs, train_loss, val_loss, train_acc, val_acc, model_name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    os.makedirs('result', exist_ok=True)
    plt.savefig(f'result/{model_name}_learning_curve.png')
    plt.close()
    print(f'[INFO] Saved learning curve plot for {model_name}')

def plot_confusion_matrix(true, pred, classes, model_name):
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')

    os.makedirs('result', exist_ok=True)
    plt.savefig(f'result/{model_name}_confusion_matrix.png')
    plt.close()
    print(f'[INFO] Saved confusion matrix for {model_name}')

def train_and_test(model_class, name, train_loader, val_loader, test_loader, device, epochs=20, lr=1e-3):
    print(f"\n[TRAINING] {name} model...")
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for ep in range(1, epochs+1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _, _ = eval_model(model, val_loader, criterion, device)
        train_loss.append(t_loss)
        val_loss.append(v_loss)
        train_acc.append(t_acc)
        val_acc.append(v_acc)
        print(f'Epoch {ep:2d}: Train Loss={t_loss:.4f} Acc={t_acc:.2f}% | Val Loss={v_loss:.4f} Acc={v_acc:.2f}%')

    test_loss, test_acc, test_targets, test_preds = eval_model(model, test_loader, criterion, device)
    print(f'[TEST] {name} Loss={test_loss:.4f} Acc={test_acc:.2f}%')
    plot_learning_curve(range(1, epochs+1), train_loss, val_loss, train_acc, val_acc, name)
    plot_confusion_matrix(test_targets, test_preds,
                          ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck'], name)

    return test_targets, test_preds

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("[INFO] Using device:", device)

    train_loader, val_loader, test_loader = create_loaders()

    mlp_true, mlp_pred = train_and_test(MLP, 'MLP', train_loader, val_loader, test_loader, device)
    cnn_true, cnn_pred = train_and_test(CNN, 'CNN', train_loader, val_loader, test_loader, device)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    for ax, (true, pred, name) in zip(axs, [(mlp_true, mlp_pred, 'MLP'), (cnn_true, cnn_pred, 'CNN')]):
        cm = confusion_matrix(true, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    os.makedirs('result', exist_ok=True)
    plt.savefig('result/combined_confusion_matrix.png')
    plt.close()
    print('[INFO] Saved combined confusion matrix plot')
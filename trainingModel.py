import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import cv2
import sys
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

tumor = []
healthy = []
for f in glob.iglob("./data/brain_tumor_dataset/yes/*.png"):
    img = cv2.imread(f)
    img = cv2.resize(img,(128,128))
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    tumor.append(img)

for f in glob.iglob("./data/brain_tumor_dataset/no/*.png"):
    img = cv2.imread(f)
    img = cv2.resize(img,(128,128)) 
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    healthy.append(img)

healthy = np.array(healthy)
tumor = np.array(tumor)
All = np.concatenate((healthy, tumor))

#print(healthy.shape)
#print(tumor.shape)
#print(np.random.choice(10, 5, replace=False))

def plot_random(healthy, tumor, num=5):
    healthy_imgs = healthy[np.random.choice(healthy.shape[0], num, replace=False)]
    tumor_imgs = tumor[np.random.choice(tumor.shape[0], num, replace=False)]
    
    plt.figure(figsize=(16,9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('healthy')
        plt.imshow(healthy_imgs[i])
    #plt.show()
        
    plt.figure(figsize=(16,9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('tumor')
        plt.imshow(tumor_imgs[i])
    #plt.show()

#plot_random(healthy, tumor, num=5)

#first half of training model

class MRIDataset(Dataset):
    def __init__(self, healthy_data, tumor_data):
        self.healthy_data = healthy_data
        self.tumor_data = tumor_data

    def __len__(self):
        return len(self.healthy_data) + len(self.tumor_data)

    def __getitem__(self, idx):
        if idx < len(self.healthy_data):
            img = self.healthy_data[idx]
            label = 0  # 0 for healthy
        else:
            img = self.tumor_data[idx - len(self.healthy_data)]
            label = 1  # 1 for tumor
        
        # Convert image to tensor
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, label
# Split data into train and validation sets
train_size = 0.8
train_healthy = healthy[:int(len(healthy) * train_size)]
train_tumor = tumor[:int(len(tumor) * train_size)]
val_healthy = healthy[int(len(healthy) * train_size):]
val_tumor = tumor[int(len(tumor) * train_size):]

# Create train and validation datasets using the new MRIDataset class
train_dataset = MRIDataset(train_healthy, train_tumor)
val_dataset = MRIDataset(val_healthy, val_tumor)

# Define model architecture
class MRIClassifier(nn.Module):
    def __init__(self):
        super(MRIClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model instance and optimizer
model = MRIClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define data loaders using the new MRIDataset class
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Train the model
num_epochs = 30
for epoch in range(num_epochs):
    # Train the model on the training set
    model.train()
    train_loss = 0
    train_acc = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
    train_loss /= len(train_loader.dataset)
    train_acc = train_acc.double() / len(train_loader.dataset)

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)
    val_loss /= len(val_loader.dataset)
    val_acc = val_acc.double() / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

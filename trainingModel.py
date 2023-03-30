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

tumor = []
healthy = []
for f in glob.iglob("./data/brain_tumor_dataset/yes/*.jpg"):
    img = cv2.imread(f)
    img = cv2.resize(img,(128,128))
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    tumor.append(img)

for f in glob.iglob("./data/brain_tumor_dataset/no/*.jpg"):
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

# Create train and validation datasets
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


#second half of training model
# define the labels for healthy and tumor images
healthy_labels = np.zeros(healthy.shape[0], dtype=np.int64)
tumor_labels = np.ones(tumor.shape[0], dtype=np.int64)

# concatenate the labels for all images
All_labels = np.concatenate((healthy_labels, tumor_labels))

# define a custom PyTorch dataset for loading the data
class MRIDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# define a transform to apply to the images
class MRITransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).float()
        return x

# define the mean and standard deviation of the image dataset
mean = np.mean(All / 255.0)
std = np.std(All / 255.0)

# define the batch size for the data loader
batch_size = 32

# create a custom PyTorch dataset for loading the data
dataset = MRIDataset(All, All_labels, transform=MRITransform(mean, std))

# split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# define the model architecture
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(128 * 16 * 16, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2),
)

# define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# define the device to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# move the model to the device
model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # get the first image in the batch
        img = images[0]
        # denormalize image tensor
        img = img * 0.5 + 0.5
        # convert image tensor to numpy array
        img = img.cpu().numpy()
        # transpose image array to match the shape required by imshow
        img = np.transpose(img, (1, 2, 0))
        
        # get the predicted label for the first image in the batch
        pred_label = predicted[0].item()
        # get the correct label for the first image in the batch
        true_label = labels[0].item()

        # show the image, predicted label, and correct label
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f"Predicted: {pred_label}, True: {true_label}")
        plt.show()


accuracy = 100 * correct / total
print('Accuracy of the network on the test images: %d %%' % accuracy)


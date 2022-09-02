#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from torch import optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create PyTorch Dataset
class RPS_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.from_numpy(np.array(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

dataset = RPS_Dataset(annotations_file="rps.csv",
                      img_dir="rps_images",
                      transform=transforms.Resize(28))

dataset_len = len(dataset) // 2
train_set, test_set = random_split(dataset, [dataset_len, dataset_len])

loaders = {
    'train' : torch.utils.data.DataLoader(train_set,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1),

    'test'  : torch.utils.data.DataLoader(test_set,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1),
}

# Creating the CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Fully connected layer, output 3 classes
        self.out = nn.Linear(32 * 7 * 7, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        # Return x for visualization
        return output, x

def train(num_epochs, cnn, loaders):
    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # Gives batch data, normalize x when iterate train_loader
            # Batch x
            b_x = Variable(images.float())
            # Batch y
            b_y = Variable(labels)

            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # Clear gradients for this training step
            optimizer.zero_grad()

            # Backpropagation, compute gradients
            loss.backward()

            # Apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                     .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test():
    # Test the model
    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images.float())
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))

        print("Test Accuracy of the model on the test images: {:.2f}".format(accuracy))

if __name__ == "__main__":
    # Defines the CNN
    cnn = CNN()

    # Optimization function
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

    # Loss function
    loss_func = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 10

    # MAYBE, MAYBE, WORKS ??!
    train(num_epochs, cnn, loaders)

    # MAYBE, MAYBE, WORKS ??!
    test()

    # Saves the model
    torch.save(cnn.state_dict(), os.getcwd()+"/rps_recognition.model")

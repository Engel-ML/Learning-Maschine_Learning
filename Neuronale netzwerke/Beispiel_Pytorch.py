#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:01:44 2024

@author: colinengelmann
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Hyperparameter
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Datentransformationen: Normalisierung auf Mittelwert und Varianz
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST Datensätze laden
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Einfaches CNN-Modell definieren
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution: Eingabe: 1-Kanal (grau), Ausgabe: 32 Features, Kernel: 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduziert die Dimension um Hälfte

        # Nach 2x Pooling von 28x28 zu 7x7 (da zwei Pooling-Operationen: 28->14->7)
        # 64 Kanäle * 7 * 7 Pixel nach letztem Conv/Pool
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Erster Conv-Block
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))  # Nach conv2 + pool: 28->14
        
        # Zweiter Conv-Block (optional, hier aber weggelassen)
        # Hier könnte man noch weitere Convs hinzufügen, aber wir lassen es einfach.

        # Noch ein Pooling, um auf 7x7 zu kommen
        # Um auf 7x7 zu kommen, ist ein weiterer Pool-Vorgang nötig:
        # (Optional: hier noch ein Conv-Layer + pool einfügen, falls man möchte.)
        
        # Da wir zwei Poolings brauchen, um von 28->14->7 zu kommen,
        # fügen wir noch einen Layer wie oben hinzu:
        
        # Alternative: Man könnte direkt zwei Poolings im forward aufeinander anwenden:
        # x = self.pool(x) # Damit auf 7x7 reduziert wird
        # Hier für Klarheit nochmal pool auf dasselbe Ergebnis:
        
        # Eigentlich könnte man nach conv2 schon auf 14x14 sein. 
        # Wir machen einfach einen weiteren conv-block um auf 7x7 zu kommen.
        # Aus Vereinfachungsgründen fügen wir noch einen conv-block hinzu:
        
        # Nehmen wir stattdessen ein Flatten hier:
        # Da obiges Netzwerk nur einmal pooling enthielt, ändern wir es leicht:
        # Wir machen zwei Conv-Ebenen mit jeweils einem Pooling:
        
        # Also final so:
        # x = self.relu(self.conv1(x))
        # x = self.pool(self.relu(self.conv2(x)))  # Jetzt 14x14
        # Fügen wir einfach noch einen dritten Conv-Layer hinzu:
        # (Um Unsicherheiten zu vermeiden, passe ich das Modell oben an.)

        # ---- Anpassung des Modells, um keine Verwirrung zu stiften ----
        # Wir definieren das Modell einfach neu mit zwei Pooling Operationen:
        # Zur Übersichtlichkeit schreibe ich das Modell hier neu:

        pass

# Neues Modell (angepasst mit genau zwei Pooling-Schritten)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Halbiert die Größe
        # Nach pool: 28x28 -> 14x14

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        # Noch ein Pooling
        # Nach diesem Pooling: 14x14 -> 7x7

        self.relu = nn.ReLU()

        # Fully Connected
        # Nach dem zweiten Pooling: Featuremap-Größe: 128 * 7 * 7
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))  # 1. Pooling: 28->14

        # Block 2
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))  # 2. Pooling: 14->7

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Device wählen (GPU falls vorhanden)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Loss und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        # Gradients zurücksetzen
        optimizer.zero_grad()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation auf dem Testset
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

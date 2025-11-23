import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
import json

# --- Визначення моделі ---
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Adaptive pooling -> фіксований розмір перед FC
        self.gap = nn.AdaptiveAvgPool2d((8, 4))
        self.fc1 = nn.Linear(64 * 8 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)  # робимо фіксований розмір
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x


# --- Параметри ---
target_classes = ['yes', 'no', 'up', 'down']
num_classes = len(target_classes)
batch_size = 32
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Перетворення ---
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

def label_to_index(word):
    return torch.tensor(target_classes.index(word))

# --- Collate function ---
def collate_fn(batch):
    tensors, targets = [], []
    max_len = max(x[0].shape[1] for x in batch)  # максимальна довжина хвилі
    
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        # паддінг waveform до max_len
        if waveform.shape[1] < max_len:
            pad_size = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # перетворення в мел-спектрограму
        spec = mel_spectrogram(waveform).squeeze(0)  # [64, time]
        tensors.append(spec)
        targets.append(label_to_index(label))  # мапимо слово в індекс
    
    # додаємо вимір "канал"
    return torch.stack(tensors).unsqueeze(1), torch.stack(targets)


# --- ОБМЕЖЕНЕ завантаження даних ---
def get_limited_dataset(subset, samples_per_class=500):
    dataset = SPEECHCOMMANDS(root="./data", download=True, subset=subset)
    
    class_counts = {cls: 0 for cls in target_classes}
    selected_indices = []
    
    for idx in range(len(dataset)):
        waveform, sample_rate, label, speaker_id, utterance_number = dataset[idx]
        
        if label in target_classes and class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
            
        if all(count >= samples_per_class for count in class_counts.values()):
            break
    
    print(f"Selected {len(selected_indices)} samples for {subset} set")
    print(f"Class distribution: {class_counts}")
    
    return Subset(dataset, selected_indices)


print("Loading limited datasets...")
train_set = get_limited_dataset('training', samples_per_class=400)
test_set = get_limited_dataset('testing', samples_per_class=100)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# --- Ініціалізація моделі, критерію, оптимізатора ---
model = AudioClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Цикл навчання ---
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 50 == 49:
            avg_loss = running_loss / 50
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

# --- Оцінка на тестовому наборі ---
print("Evaluating on test set...")
model.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
avg_test_loss = test_loss / len(test_loader)

print(f'Test Results:')
print(f'Accuracy: {test_accuracy:.2f}%')
print(f'Average Loss: {avg_test_loss:.4f}')

# --- Збереження ---
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")

with open('class_info.json', 'w') as f:
    json.dump({
        'target_classes': target_classes
    }, f)

print("Training completed successfully!")

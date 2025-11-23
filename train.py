import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
import json

# --- Model definition ---
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # Reduced channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Reduced dropout
        self.gap = nn.AdaptiveAvgPool2d((4, 2))  # Smaller pooling
        self.fc1 = nn.Linear(32 * 4 * 2, 64)  # Reduced dimensions
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x

# --- Параметри ---
target_classes = ['yes', 'no', 'up', 'down']
num_classes = len(target_classes)
batch_size = 8  # Reduced batch size
epochs = int(os.getenv('EPOCHS', '2'))
samples_per_class = int(os.getenv('SAMPLES_PER_CLASS', '10'))  # Minimal samples
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using {samples_per_class} samples per class")

# --- Спрощене перетворення ---
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,  # Reduced FFT
    hop_length=256,
    n_mels=32  # Reduced mel bands
)

def label_to_index(word):
    return torch.tensor(target_classes.index(word))

def simple_collate_fn(batch):
    tensors, targets = [], []
    
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        # Simple processing without padding to save memory
        spec = mel_spectrogram(waveform).squeeze(0)
        # Take only first 32 time frames
        spec = spec[:, :32] if spec.shape[1] > 32 else spec
        tensors.append(spec)
        targets.append(label_to_index(label))
    
    return torch.stack(tensors).unsqueeze(1), torch.stack(targets)

# --- Мінімальний dataset ---
def get_minimal_dataset(subset, samples_per_class=10):
    print(f"🔄 Loading minimal {subset} dataset ({samples_per_class} samples per class)...")
    
    try:
        dataset = SPEECHCOMMANDS(root="./data", download=True, subset=subset)
        
        class_counts = {cls: 0 for cls in target_classes}
        selected_indices = []
        
        # Швидкий пошук необхідних класів
        for idx in range(min(1000, len(dataset))):  # Limit search
            try:
                _, _, label, _, _ = dataset[idx]
                if label in target_classes and class_counts[label] < samples_per_class:
                    selected_indices.append(idx)
                    class_counts[label] += 1
                    
                if all(count >= samples_per_class for count in class_counts.values()):
                    break
            except:
                continue
        
        print(f"✅ Selected {len(selected_indices)} samples")
        return Subset(dataset, selected_indices)
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        # Synthetic fallback
        from torch.utils.data import TensorDataset
        print("🎲 Using synthetic data")
        num_samples = samples_per_class * len(target_classes)
        dummy_inputs = torch.randn(num_samples, 1, 32, 32)  # Smaller inputs
        dummy_labels = torch.randint(0, len(target_classes), (num_samples,))
        return TensorDataset(dummy_inputs, dummy_labels)

print("📥 Loading minimal datasets...")
train_set = get_minimal_dataset('training', samples_per_class)
test_set = get_minimal_dataset('testing', samples_per_class//2)  # Even smaller test set

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=simple_collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=simple_collate_fn)

print(f"📊 Train: {len(train_loader)} batches, Test: {len(test_loader)} batches")

# --- Обмежене тренування ---
model = AudioClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🚀 Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= 5:  # Limit to 5 batches per epoch
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/5:.4f}')

# --- Швидка оцінка ---
print("📈 Evaluating...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        if i >= 3:  # Limit test batches
            break
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total if total > 0 else 0
print(f'✅ Test Accuracy: {accuracy:.2f}%')

# --- Збереження ---
torch.save(model.state_dict(), 'model.pth')
print("💾 Model saved")

with open('class_info.json', 'w') as f:
    json.dump({'target_classes': target_classes}, f)

with open('training.log', 'w') as f:
    f.write(f"Minimal training completed. Accuracy: {accuracy:.2f}%\n")

print("🎉 Training completed successfully!")
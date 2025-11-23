import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import glob

# --- Кастомний Dataset для вашої структури ---
class CustomSpeechCommands(Dataset):
    def __init__(self, data_dir, classes, subset='training'):
        self.data_dir = data_dir
        self.classes = classes
        self.subset = subset
        self.filepaths = []
        self.labels = []
        
        # Шлях до ваших даних
        base_path = os.path.join(data_dir, 'SpeechCommands', 'speech_commands_v0.02')
        
        for class_name in classes:
            # Шукаємо всі аудіо файли для цього класу
            pattern = os.path.join(base_path, class_name, '*.wav')
            files = glob.glob(pattern)
            
            for file_path in files:
                self.filepaths.append(file_path)
                self.labels.append(class_name)
        
        print(f"📁 Завантажено {len(self.filepaths)} файлів для {subset}")
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        label = self.labels[idx]
        
        try:
            # Завантажуємо аудіо файл
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate, label, "speaker_0", 0
        except Exception as e:
            print(f"❌ Помилка завантаження {file_path}: {e}")
            # Повертаємо dummy data у разі помилки
            dummy_audio = torch.zeros(1, 16000)
            return dummy_audio, 16000, label, "speaker_0", 0

# --- Параметри ---
target_classes = ['yes', 'no', 'up', 'down']
num_classes = len(target_classes)
batch_size = 16  # Зменшено для CI/CD
epochs = 2
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
    
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        # Перетворення в мел-спектрограму
        spec = mel_spectrogram(waveform).squeeze(0)  # [64, time]
        tensors.append(spec)
        targets.append(label_to_index(label))
    
    # Знаходимо максимальну довжину для padding
    max_time = max(spec.shape[1] for spec in tensors)
    
    # Padding всіх спектрограм до однакової довжини
    padded_tensors = []
    for spec in tensors:
        if spec.shape[1] < max_time:
            pad_size = max_time - spec.shape[1]
            spec = torch.nn.functional.pad(spec, (0, pad_size))
        padded_tensors.append(spec)
    
    return torch.stack(padded_tensors).unsqueeze(1), torch.stack(targets)

# --- Завантаження даних ---
def get_limited_dataset(subset, samples_per_class=50):
    """Завантажує дані з вашої структури папок"""
    dataset = CustomSpeechCommands('./data', target_classes, subset=subset)
    
    # Обмежуємо кількість зразків для швидшого тренування
    if samples_per_class < len(dataset):
        # Просто беремо перші N зразків
        indices = list(range(min(samples_per_class * len(target_classes), len(dataset))))
        from torch.utils.data import Subset
        return Subset(dataset, indices)
    
    return dataset

print("Loading datasets...")
train_set = get_limited_dataset('training', samples_per_class=50)
test_set = get_limited_dataset('testing', samples_per_class=20)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
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

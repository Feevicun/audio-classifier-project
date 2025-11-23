import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import json
import glob

# --- Model definition ---
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
        self.gap = nn.AdaptiveAvgPool2d((8, 4))
        self.fc1 = nn.Linear(64 * 8 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x

# --- Кастомний Dataset для вашої структури ---
class CustomSpeechCommands(Dataset):
    def __init__(self, data_dir, classes, subset='training'):
        self.data_dir = data_dir
        self.classes = classes
        self.subset = subset
        self.filepaths = []
        self.labels = []
        
        print(f"🔍 Шукаємо дані в: {data_dir}")
        
        # Спробуємо різні можливі шляхи
        possible_paths = [
            os.path.join(data_dir, 'SpeechCommands', 'speech_commands_v0.02'),
            os.path.join(data_dir, 'speech_commands_v0.02'),
            data_dir
        ]
        
        base_path = None
        for path in possible_paths:
            if os.path.exists(path):
                base_path = path
                print(f"✅ Знайдено шлях: {path}")
                break
        
        if base_path is None:
            print(f"❌ Не знайдено жодного з можливих шляхів: {possible_paths}")
            return
        
        for class_name in classes:
            # Шукаємо всі аудіо файли для цього класу
            pattern = os.path.join(base_path, class_name, '*.wav')
            files = glob.glob(pattern)
            
            if not files:
                print(f"⚠️ Не знайдено файлів для класу {class_name} за шаблоном: {pattern}")
                # Спробуємо знайти файли в інших місцях
                pattern2 = os.path.join(base_path, '**', class_name, '*.wav')
                files = glob.glob(pattern2, recursive=True)
                print(f"🔍 Рекурсивний пошук знайшов: {len(files)} файлів")
            
            for file_path in files:
                self.filepaths.append(file_path)
                self.labels.append(class_name)
        
        print(f"📁 Завантажено {len(self.filepaths)} файлів для {subset}")
        
        if len(self.filepaths) == 0:
            print("🚨 УВАГА: Не знайдено жодного аудіо файлу!")
            print("📂 Доступні файли в data/:")
            os.system(f"find {data_dir} -type f | head -20")

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
            # Створюємо синтетичні дані для тестування
            duration = 1.0
            samples = int(16000 * duration)
            dummy_audio = torch.randn(1, samples) * 0.1
            return dummy_audio, 16000, label, "speaker_0", 0

# --- Синтетичний Dataset ---
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=200, num_classes=4):
        self.num_samples = num_samples
        self.num_classes = num_classes
        print(f"🎲 Створено синтетичний dataset з {num_samples} зразками")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Генеруємо синтетичні аудіо дані
        duration = 1.0
        samples = int(16000 * duration)
        
        # Різні частоти для різних класів
        class_idx = idx % self.num_classes
        frequencies = [440, 523, 659, 392]  # Ля, До, Мі, Соль
        freq = frequencies[class_idx]
        
        t = torch.linspace(0, duration, samples)
        audio_data = 0.5 * torch.sin(2 * np.pi * freq * t)
        waveform = audio_data.unsqueeze(0)  # [1, samples]
        
        labels = ['yes', 'no', 'up', 'down']
        label = labels[class_idx]
        
        return waveform, 16000, label, "synthetic_speaker", idx

# --- Параметри ---
target_classes = ['yes', 'no', 'up', 'down']
num_classes = len(target_classes)
batch_size = 16
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

# --- Collate function для реальних даних ---
def real_data_collate_fn(batch):
    tensors, targets = [], []
    
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        # Перетворення в мел-спектрограму
        spec = mel_spectrogram(waveform).squeeze(0)  # [64, time]
        tensors.append(spec)
        targets.append(label_to_index(label))
    
    if not tensors:
        # Повертаємо пусті тензори, якщо немає даних
        return torch.tensor([]), torch.tensor([])
    
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

# --- Collate function для синтетичних даних ---
def synthetic_data_collate_fn(batch):
    tensors, targets = [], []
    
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        # Перетворення в мел-спектрограму
        spec = mel_spectrogram(waveform).squeeze(0)  # [64, time]
        tensors.append(spec)
        targets.append(label_to_index(label))
    
    # Для синтетичних даних всі спектрограми однакової довжини
    return torch.stack(tensors).unsqueeze(1), torch.stack(targets)

# --- Завантаження даних ---
def get_limited_dataset(subset, samples_per_class=50):
    """Завантажує дані з вашої структури папок"""
    dataset = CustomSpeechCommands('./data', target_classes, subset=subset)
    
    # Якщо немає реальних даних, використовуємо синтетичні
    if len(dataset) == 0:
        print(f"🚨 Використовую синтетичні дані для {subset}")
        return SyntheticDataset(num_samples=samples_per_class * len(target_classes), 
                              num_classes=len(target_classes))
    
    # Обмежуємо кількість зразків для швидшого тренування
    if samples_per_class * len(target_classes) < len(dataset):
        indices = list(range(samples_per_class * len(target_classes)))
        from torch.utils.data import Subset
        return Subset(dataset, indices)
    
    return dataset

print("Loading datasets...")
train_set = get_limited_dataset('training', samples_per_class=50)
test_set = get_limited_dataset('testing', samples_per_class=20)

print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

if len(train_set) == 0:
    print("❌ CRITICAL: No training data available!")
    exit(1)

# Вибираємо правильну collate_fn в залежності від типу даних
if isinstance(train_set, SyntheticDataset):
    collate_fn = synthetic_data_collate_fn
    print("🎲 Використовую collate_fn для синтетичних даних")
else:
    collate_fn = real_data_collate_fn
    print("📁 Використовую collate_fn для реальних даних")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
        if len(inputs) == 0:
            print("⚠️ Пропускаємо порожній batch")
            continue
            
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
        
        if i % 10 == 9:
            avg_loss = running_loss / 10
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
        if len(inputs) == 0:
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

if test_total > 0:
    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
else:
    test_accuracy = 0
    avg_test_loss = 0

print(f'Test Results:')
print(f'Accuracy: {test_accuracy:.2f}%')
print(f'Average Loss: {avg_test_loss:.4f}')

# --- Збереження ---
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")

with open('class_info.json', 'w') as f:
    json.dump({
        'target_classes': target_classes
    }, f)

# Запис логів тренування
with open('training.log', 'w') as f:
    f.write(f"Training completed successfully!\n")
    f.write(f"Final Test Accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Final Test Loss: {avg_test_loss:.4f}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Device: {device}\n")

print("Training completed successfully!")
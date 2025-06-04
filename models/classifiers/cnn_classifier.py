import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_dataset(dataset_path):
    """Загрузка датасета"""
    data = np.load(dataset_path)
    x_noisy = torch.FloatTensor(data['x_noisy'])
    labels = torch.LongTensor(data['labels'])
    return x_noisy, labels

def create_dataloader(x_noisy, labels, batch_size=32):
    """Создание DataLoader для пакетной обработки"""
    dataset = torch.utils.data.TensorDataset(x_noisy, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, train_loader, val_loader, device, epochs=30):
    """Обучение модели"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch + 1}/{epochs} (train)')
        for batch_x, batch_labels in progress_bar:
            optimizer.zero_grad()
            
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            total_loss += loss.item()
            
            train_accuracy = total_correct / total_samples
            train_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'accuracy': f'{train_accuracy:.4f}'
            })
        
        # Валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Эпоха {epoch + 1}/{epochs} (val)')
            for batch_x, batch_labels in progress_bar:
                batch_x = batch_x.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == batch_labels).sum().item()
                val_samples += batch_labels.size(0)
                val_loss += loss.item()
                
                val_accuracy = val_correct / val_samples
                val_avg_loss = val_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({
                    'loss': f'{val_avg_loss:.4f}',
                    'accuracy': f'{val_accuracy:.4f}'
                })
        
        print(f'Эпоха {epoch + 1}/{epochs}:')
        print(f'Train — Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'Val — Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.4f}\n')
        
        # Сохраняем лучшую модель
        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_avg_loss,
                'accuracy': val_accuracy
            }, 'cnn_classifier_weights.pth')
        else:
            patience_counter += 1
        
        # Обновляем learning rate
        scheduler.step(val_avg_loss)
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Ранняя остановка на эпохе {epoch + 1}, валидационный лосс не улучшается {patience} эпох')
            break

def save_model(model, path):
    """Сохранение весов модели"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Загрузка весов модели"""
    model.load_state_dict(torch.load(path))
    return model

def normalize_data(x):
    """Нормализация данных по каналам"""
    # Добавляем размерность батча, если её нет
    if x.ndim == 2:
        x = x[np.newaxis, ...]  # (2, seq_len) -> (1, 2, seq_len)
    
    # x shape: (N, 2, sequence_length) или (1, 2, sequence_length)
    mean = x.mean(axis=(0, 2), keepdims=True)  # (1, 2, 1)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-8
    return (x - mean) / std

def main():
    # Параметры
    DATASET_PATH = "data/cnn_dataset.npz"  # Используем меньший датасет для CNN
    WEIGHTS_PATH = "weights/cnn_classifier_weights.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Загрузка данных
    data = np.load(DATASET_PATH)
    x_noisy = data['x_noisy']  # (N, 2, sequence_length)
    labels = data['labels']
    
    print(f"\nРазмерности данных:")
    print(f"x_noisy: {x_noisy.shape}")
    print(f"labels: {labels.shape}")
    
    # Нормализация данных
    x_noisy = normalize_data(x_noisy)
    
    # Разделение на train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_noisy, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Создание DataLoader'ов
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), 
        torch.LongTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Создание модели
    model = CNNClassifier().to(DEVICE)
    
    # Выводим количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Обучение
    train_model(model, train_loader, val_loader, DEVICE)
    
    print("Обучение завершено. Проверьте файлы training_curves.png и cnn_classifier_weights.pth")

if __name__ == "__main__":
    main() 
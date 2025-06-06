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
            # Входной сигнал теперь имеет 1 канал вместо 2
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
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
    if x_noisy.ndim == 2:  # Если данные одномерные
        x_noisy = x_noisy.unsqueeze(1)  # Добавляем размерность канала
    labels = torch.LongTensor(data['labels'])
    return x_noisy, labels

def create_dataloader(x_noisy, labels, batch_size=32):
    """Создание DataLoader для пакетной обработки"""
    dataset = torch.utils.data.TensorDataset(x_noisy, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def augment_batch(batch_x):
    # Случайный гауссовский шум
    if torch.rand(1).item() < 0.7:
        noise = torch.randn_like(batch_x) * 0.05
        batch_x = batch_x + noise
    # Случайное масштабирование амплитуды
    if torch.rand(1).item() < 0.7:
        scale = torch.rand(batch_x.size(0), 1, 1, device=batch_x.device) * 0.4 + 0.8
        batch_x = batch_x * scale
    return batch_x

def train_and_validate(model, train_loader, val_loader, device, class_weights=None, epochs=20, weights_path=None):
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    print("\nStarting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f'Эпоха {epoch+1}/{epochs} (train)', leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = augment_batch(batch_x)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        # Валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f'Эпоха {epoch+1}/{epochs} (val)', leave=False):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        print(f'Эпоха {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if weights_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs
                }, weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping на эпохе {epoch+1}')
                break

def save_model(model, path):
    """Сохранение весов модели"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Загрузка весов модели"""
    model.load_state_dict(torch.load(path))
    return model

def normalize_data(x):
    """Нормализация данных"""
    # Добавляем размерность батча, если её нет
    if x.ndim == 1:
        x = x[np.newaxis, np.newaxis, :]  # (seq_len,) -> (1, 1, seq_len)
    elif x.ndim == 2:
        x = x[:, np.newaxis, :]  # (batch, seq_len) -> (batch, 1, seq_len)
    
    # x shape: (N, 1, sequence_length) или (1, 1, sequence_length)
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True) + 1e-8
    return (x - mean) / std

def main():
    # Параметры
    DATASET_PATH = "datasets/cnn_dataset.npz"  # Обновленный путь к датасету
    WEIGHTS_PATH = "weights/cnn_classifier_weights.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    
    print("Starting CNN classifier training...")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nLoading dataset from {DATASET_PATH}...")
    try:
        # Загрузка данных
        x_noisy, labels = load_dataset(DATASET_PATH)
        
        print(f"\nРазмерности данных:")
        print(f"x_noisy: {x_noisy.shape}")
        print(f"labels: {labels.shape}")
        
        # Разделение на train/val
        print("\nSplitting dataset into train/val...")
        x_train, x_val, y_train, y_val = train_test_split(
            x_noisy, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        print(f"Train size: {len(x_train)}")
        print(f"Val size: {len(x_val)}")
        
        # Создание DataLoader'ов
        print("\nCreating DataLoaders...")
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        
        # Создание и перемещение модели на GPU
        print("\nInitializing model...")
        model = CNNClassifier().to(DEVICE)
        
        # Вывод информации о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Обучение модели
        print("\nStarting training...")
        train_and_validate(model, train_loader, val_loader, DEVICE, epochs=20, weights_path=WEIGHTS_PATH)
        
        # Сохранение весов
        print(f"\nSaving model weights to {WEIGHTS_PATH}...")
        save_model(model, WEIGHTS_PATH)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_and_analyze_data(dataset_path):
    """Загрузка и анализ данных"""
    print("Анализ данных...")
    data = np.load(dataset_path)
    x_noisy = data['x_noisy']
    labels = data['labels']
    
    print(f"\nРазмерности данных:")
    print(f"x_noisy: {x_noisy.shape}")
    print(f"labels: {labels.shape}")
    
    print(f"\nСтатистика данных:")
    print(f"min: {x_noisy.min():.4f}")
    print(f"max: {x_noisy.max():.4f}")
    print(f"mean: {x_noisy.mean():.4f}")
    print(f"std: {x_noisy.std():.4f}")
    
    print(f"\nРаспределение классов:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"Класс {cls}: {count} примеров ({count/len(labels)*100:.2f}%)")
    
    # Визуализация примеров
    plt.figure(figsize=(15, 5))
    for i in range(7):  # По одному примеру каждого класса
        idx = np.where(labels == i)[0][0]
        plt.subplot(2, 4, i+1)
        plt.plot(x_noisy[idx, 0], label='I')
        plt.plot(x_noisy[idx, 1], label='Q')
        plt.title(f'Класс {i}')
        plt.legend()
    plt.tight_layout()
    plt.savefig('signal_examples.png')
    plt.close()
    
    return x_noisy, labels

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Определим размер входа для полносвязного слоя динамически
        with torch.no_grad():
            x = torch.randn(1, 2, 512)
            x = self.features(x)
            n_features = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_and_validate(model, train_loader, val_loader, device, epochs=30):
    """Обучение модели с валидацией"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Эпоха {epoch+1}/{epochs} (train)'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
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
            for batch_x, batch_y in tqdm(val_loader, desc=f'Эпоха {epoch+1}/{epochs} (val)'):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f'Эпоха {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping на эпохе {epoch+1}')
                break
    
    # Построение графиков
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def main():
    DATASET_PATH = "cnn_dataset.npz"  # или "full_dataset.npz"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    
    # Загрузка и анализ данных
    x_noisy, labels = load_and_analyze_data(DATASET_PATH)
    
    # Преобразование в тензоры
    x_noisy = torch.FloatTensor(x_noisy)
    labels = torch.LongTensor(labels)
    
    # Разделение на train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_noisy, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Создание DataLoader'ов
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Создание и обучение модели
    model = SimpleCNN().to(DEVICE)
    train_and_validate(model, train_loader, val_loader, DEVICE)
    
    print("Диагностика завершена. Проверьте файлы signal_examples.png и training_curves.png")

if __name__ == "__main__":
    main() 
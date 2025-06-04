import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TransformerClassifier(nn.Module):
    def __init__(self, sequence_length=512, input_dim=2, d_model=128, nhead=4, num_layers=2, num_classes=7):
        super(TransformerClassifier, self).__init__()
        
        # Входной слой с нормализацией
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Проекция входа
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=1),
            nn.ReLU()
        )
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        # Transformer энкодер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,  # Уменьшаем размер feed-forward слоя
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.Sequential(
            encoder_layer,
            nn.LayerNorm(d_model),
            encoder_layer,
            nn.LayerNorm(d_model)
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(d_model * sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 2, sequence_length)
        
        # Нормализация входа
        x = self.input_norm(x)
        
        # Проекция входа
        x = self.input_projection(x)  # (batch_size, d_model, sequence_length)
        
        # Перестановка для transformer
        x = x.transpose(1, 2)  # (batch_size, sequence_length, d_model)
        
        # Позиционное кодирование
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)
        
        # Подготовка для классификатора
        x = x.reshape(x.size(0), -1)  # (batch_size, sequence_length * d_model)
        
        # Классификация
        x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=513, dropout=0.1):  # 513 = 512 + 1 для CLS токена
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def normalize_data(x):
    """Нормализация данных по каналам"""
    # Добавляем размерность батча, если её нет
    if x.ndim == 2:
        x = x[np.newaxis, ...]  # (2, seq_len) -> (1, 2, seq_len)
    
    # x shape: (N, 2, sequence_length) или (1, 2, sequence_length)
    mean = x.mean(axis=(0, 2), keepdims=True)  # (1, 2, 1)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-8
    return (x - mean) / std

def compute_class_weights(labels):
    """Вычисление весов классов для борьбы с дисбалансом"""
    counts = Counter(labels.numpy())
    total = len(labels)
    weights = torch.zeros(len(counts))
    for cls, count in counts.items():
        weights[cls] = total / (len(counts) * count)
    return weights

def load_and_analyze_data(dataset_path):
    """Загрузка и анализ данных"""
    print("Анализ данных...")
    data = np.load(dataset_path)
    x_noisy = data['x_noisy']  # (N, 2, sequence_length)
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
    
    return torch.FloatTensor(x_noisy), torch.LongTensor(labels)

def train_and_validate(model, train_loader, val_loader, device, class_weights=None, epochs=30):
    """Обучение модели с валидацией"""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Планировщик с разогревом и косинусным затуханием
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 10
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
            
            # Выводим текущий learning rate
            if train_total % 1000 == 0:
                print(f'Current lr: {scheduler.get_last_lr()[0]:.2e}')
        
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
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'transformer_classifier_weights.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping на эпохе {epoch+1}')
                break
    
    # Построение графиков
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    
    # График learning rate
    plt.subplot(1, 3, 3)
    steps = np.arange(num_training_steps)
    lrs = [lr_lambda(step) * 2e-4 for step in steps]
    plt.plot(steps, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_curves_transformer.png')
    plt.close()

def main():
    # Параметры
    DATASET_PATH = "data/full_dataset.npz"
    WEIGHTS_PATH = "weights/transformer_classifier_weights.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 30
    
    print("Модель: Transformer Classifier")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Создание модели
    model = TransformerClassifier().to(DEVICE)
    
    # Загрузка данных
    x_noisy, labels = load_and_analyze_data(DATASET_PATH)
    
    # Вычисляем веса классов
    class_weights = compute_class_weights(labels)
    print("\nВеса классов:")
    for i, w in enumerate(class_weights):
        print(f"Класс {i}: {w:.4f}")
    
    # Разделение на train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_noisy, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Создание DataLoader'ов
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Обучение модели
    train_and_validate(model, train_loader, val_loader, DEVICE, class_weights=class_weights)
    
    print("Обучение завершено. Проверьте файлы training_curves_transformer.png и transformer_classifier_weights.pth")

if __name__ == "__main__":
    main() 
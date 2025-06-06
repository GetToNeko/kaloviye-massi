import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.utils.checkpoint
import torch.cuda.amp
import math

class TransformerClassifier(nn.Module):
    def __init__(self, sequence_length=512, d_model=128, nhead=8, num_layers=3, num_classes=7, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        # Проекция входного сигнала
        self.input_projection = nn.Linear(1, d_model)
        # Позиционное кодирование
        self.pos_encoder = nn.Parameter(torch.randn(1, sequence_length, d_model))
        self.dropout = nn.Dropout(dropout)
        # Transformer энкодер
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation='gelu'
            )
            encoder_layers.append(encoder_layer)
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        # Классификатор
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.zeros_(p)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, 1)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoder
        x = self.dropout(x)
        for layer in self.transformer_encoder:
            x = layer(x)
        x = x.mean(dim=1)  # mean pooling
        x = self.norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=513, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def normalize_data(x):
    """Универсальная нормализация данных для одноканального сигнала"""
    # numpy
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x[np.newaxis, np.newaxis, :]
        elif x.ndim == 2:
            x = x[:, np.newaxis, :]
        mean = x.mean(axis=2, keepdims=True)
        std = x.std(axis=2, keepdims=True) + 1e-8
        return (x - mean) / std
    # torch.Tensor
    elif isinstance(x, torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        return (x - mean) / std
    else:
        raise ValueError("normalize_data: input must be np.ndarray or torch.Tensor")

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
    x_noisy = data['x_noisy']
    if x_noisy.ndim == 2:  # Если данные одномерные
        x_noisy = x_noisy[:, np.newaxis, :]  # Добавляем размерность канала
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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Создает scheduler с косинусным затуханием и warmup периодом
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Косинусное затухание
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_and_validate(model, train_loader, val_loader, device, class_weights=None, epochs=30, weights_path=None):
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
    plt.tight_layout()
    plt.savefig('training_curves_transformer.png')
    plt.close()

def main():
    # Параметры
    DATASET_PATH = "datasets/transformer_dataset.npz"  # Возвращаем датасет для трансформера
    WEIGHTS_PATH = "weights/transformer_classifier_weights_v2.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64  # Увеличенный размер батча
    
    print("Starting Transformer classifier training...")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        print(f"\nLoading dataset from {DATASET_PATH}...")
        # Загрузка и анализ данных
        x_noisy, labels = load_and_analyze_data(DATASET_PATH)
        
        print(f"\nРазмерности данных:")
        print(f"x_noisy: {x_noisy.shape}")
        print(f"labels: {labels.shape}")
        
        # Вычисление весов классов для борьбы с дисбалансом
        print("\nComputing class weights...")
        class_weights = compute_class_weights(labels)
        print(f"Class weights: {class_weights}")
        
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
        model = TransformerClassifier().to(DEVICE)
        
        # Вывод информации о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Обучение модели
        print("\nStarting training...")
        train_and_validate(model, train_loader, val_loader, DEVICE, class_weights, epochs=30, weights_path=WEIGHTS_PATH)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during training:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
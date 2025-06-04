import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

class TransformerDenoiser(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):  # Уменьшаем размеры модели
        super(TransformerDenoiser, self).__init__()
        
        # Входной слой
        self.input_layer = nn.Linear(2, d_model)
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer энкодер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,  # Уменьшаем размер feed-forward слоя
            batch_first=True  # Важно для эффективности
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(d_model, 2)
        
    def forward(self, x):
        # x shape: (batch_size, 2, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 2)
        
        # Преобразование входа
        x = self.input_layer(x)  # (batch_size, sequence_length, d_model)
        
        # Добавление позиционного кодирования
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Выходной слой
        x = self.output_layer(x)  # (batch_size, sequence_length, 2)
        
        # Возвращаем к исходной форме
        x = x.permute(0, 2, 1)  # (batch_size, 2, sequence_length)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Добавляем размерность batch
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def load_dataset(dataset_path):
    """Загрузка датасета"""
    data = np.load(dataset_path)
    x_clean = torch.FloatTensor(data['x_clean'])
    x_noisy = torch.FloatTensor(data['x_noisy'])
    return x_clean, x_noisy

def create_dataloader(x_clean, x_noisy, batch_size=32):
    """Создание DataLoader для пакетной обработки"""
    dataset = torch.utils.data.TensorDataset(x_clean, x_noisy)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, x_clean, x_noisy, device, epochs=30, batch_size=32):
    """Обучение модели"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Создаем DataLoader
    train_loader = create_dataloader(x_clean, x_noisy, batch_size)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        # Используем tqdm для отображения прогресса
        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch + 1}/{epochs}')
        
        for batch_clean, batch_noisy in progress_bar:
            optimizer.zero_grad()
            
            # Перемещаем данные на GPU
            batch_clean = batch_clean.to(device)
            batch_noisy = batch_noisy.to(device)
            
            # Прямой проход
            output = model(batch_noisy)
            loss = criterion(output, batch_clean)
            
            # Обратное распространение
            loss.backward()
            optimizer.step()
            
            # Обновляем статистику
            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        print(f'Эпоха {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}')
        
        # Обновляем learning rate на основе функции потерь
        scheduler.step(avg_loss)
        
        # Проверяем улучшение
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Если лосс не улучшается patience эпох подряд, останавливаем обучение
        if patience_counter >= patience:
            print(f'Ранняя остановка на эпохе {epoch + 1}, лосс не улучшается {patience} эпох')
            break

def save_model(model, path):
    """Сохранение весов модели"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Загрузка весов модели"""
    model.load_state_dict(torch.load(path))
    return model

def main():
    # Параметры
    DATASET_PATH = "full_dataset.npz"
    WEIGHTS_PATH = "transformer_denoiser.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32  # Размер пакета
    EPOCHS = 30     # Увеличиваем количество эпох
    
    print("Модель: Transformer Denoiser")
    
    # Создание модели
    model = TransformerDenoiser().to(DEVICE)
    
    # Проверка существования весов
    if os.path.exists(WEIGHTS_PATH):
        response = input("Файл весов найден. Переобучить модель? (y/n): ")
        if response.lower() != 'y':
            print("Complete")
            return
    
    # Загрузка данных
    x_clean, x_noisy = load_dataset(DATASET_PATH)
    
    # Очищаем кэш CUDA перед обучением
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Обучение модели
    train_model(model, x_clean, x_noisy, DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Сохранение весов
    save_model(model, WEIGHTS_PATH)
    print("Complete")

if __name__ == "__main__":
    main() 

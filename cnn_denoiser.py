import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

class CNNDenoiser(nn.Module):
    def __init__(self):
        super(CNNDenoiser, self).__init__()
        
        # Энкодер (уменьшаем количество фильтров)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 2, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    DATASET_PATH = "cnn_dataset.npz"
    WEIGHTS_PATH = "cnn_denoiser.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32  # Размер пакета
    EPOCHS = 30     # Увеличиваем количество эпох
    
    print("Модель: CNN Denoiser")
    
    # Создание модели
    model = CNNDenoiser().to(DEVICE)
    
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

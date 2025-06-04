import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def train_model(model, train_loader, val_loader, device, epochs=30):
    """Обучение модели"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch + 1}/{epochs} (train)')
        for batch_clean, batch_noisy in progress_bar:
            optimizer.zero_grad()
            
            batch_clean = batch_clean.to(device)
            batch_noisy = batch_noisy.to(device)
            
            output = model(batch_noisy)
            loss = criterion(output, batch_clean)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            avg_train_loss = total_train_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({'loss': f'{avg_train_loss:.4f}'})
        
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Эпоха {epoch + 1}/{epochs} (val)')
            for batch_clean, batch_noisy in progress_bar:
                batch_clean = batch_clean.to(device)
                batch_noisy = batch_noisy.to(device)
                
                output = model(batch_noisy)
                loss = criterion(output, batch_clean)
                
                total_val_loss += loss.item()
                avg_val_loss = total_val_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({'loss': f'{avg_val_loss:.4f}'})
        
        val_losses.append(avg_val_loss)
        
        print(f'Эпоха {epoch + 1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}\n')
        
        # Обновляем learning rate
        scheduler.step(avg_val_loss)
        
        # Проверяем улучшение
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'weights/cnn_denoiser.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Ранняя остановка на эпохе {epoch + 1}, лосс не улучшается {patience} эпох')
            break
    
    # Построение графиков
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('CNN Denoiser - Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_denoiser_training.png')
    plt.close()

def main():
    # Параметры
    DATASET_PATH = "data/cnn_dataset.npz"
    WEIGHTS_PATH = "weights/cnn_denoiser.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 30
    
    print("Модель: CNN Denoiser")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    # Разделение на train и validation
    x_clean_train, x_clean_val, x_noisy_train, x_noisy_val = train_test_split(
        x_clean, x_noisy, test_size=0.2, random_state=42
    )
    
    # Создание DataLoader'ов
    train_loader = create_dataloader(x_clean_train, x_noisy_train, BATCH_SIZE)
    val_loader = create_dataloader(x_clean_val, x_noisy_val, BATCH_SIZE)
    
    # Очищаем кэш CUDA перед обучением
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Обучение модели
    train_model(model, train_loader, val_loader, DEVICE, epochs=EPOCHS)
    
    print("Complete")

if __name__ == "__main__":
    main() 
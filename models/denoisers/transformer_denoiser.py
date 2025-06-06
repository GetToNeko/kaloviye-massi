import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scripts.interactive_modulation import generate_modulated_signal
from scripts.visualization import plot_signals, plot_spectra, compute_snr

class TransformerDenoiser(nn.Module):
    def __init__(self, sequence_length=512, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDenoiser, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, sequence_length, d_model))
        self.dropout = nn.Dropout(dropout)
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
        self.output_layer = nn.Linear(d_model, 1)
        self._init_weights()
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.zeros_(p)
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 1)
        x = self.input_projection(x)
        x = x + self.pos_encoder
        x = self.dropout(x)
        for layer in self.transformer_encoder:
            x = layer(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.output_layer(x)  # (batch_size, sequence_length, 1)
        x = x.permute(0, 2, 1)  # (batch_size, 1, sequence_length)
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
    # Проверяем размерности и добавляем канал если нужно
    if x_clean.ndim == 2:
        x_clean = x_clean.unsqueeze(1)
    if x_noisy.ndim == 2:
        x_noisy = x_noisy.unsqueeze(1)
    # Приводим к одноканальному формату, если каналов больше одного
    if x_clean.shape[1] > 1:
        x_clean = x_clean[:, :1, :]
    if x_noisy.shape[1] > 1:
        x_noisy = x_noisy[:, :1, :]
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
            }, 'transformer_denoiser.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Ранняя остановка на эпохе {epoch + 1}, лосс не улучшается {patience} эпох')
            break
    
    # Построение графиков
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Transformer Denoiser - Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_denoiser_training.png')
    plt.close()

def interactive_test():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WEIGHTS_PATH = "weights/transformer_denoiser.pth"
    # Загрузка модели
    model = TransformerDenoiser().to(DEVICE)
    if not os.path.exists(WEIGHTS_PATH):
        print("Веса не найдены! Сначала обучите модель.")
        return
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}\nПроверьте, что файл содержит только state_dict или dict с ключом 'model_state_dict'.")
        return
    model.eval()
    # Меню выбора модуляции и SNR
    mod_types = ['AM', 'FM', 'PM', 'ASK', 'FSK', 'BPSK', 'QPSK']
    print("\nВыберите тип модуляции:")
    for i, m in enumerate(mod_types, 1):
        print(f"{i}) {m}")
    while True:
        mod_choice = input("Введите номер модуляции: ").strip()
        if mod_choice in map(str, range(1, 8)):
            mod_type = mod_types[int(mod_choice)-1]
            break
        print("Неверный выбор.")
    snr_levels = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    print("\nВыберите уровень шума (SNR дБ):")
    for i, snr in enumerate(snr_levels, 1):
        print(f"{i}) {snr} дБ")
    while True:
        snr_choice = input("Введите номер уровня шума: ").strip()
        if snr_choice in map(str, range(1, 10)):
            snr = snr_levels[int(snr_choice)-1]
            break
        print("Неверный выбор.")
    # Генерация сигнала
    message, clean_signal, noisy_signal = generate_modulated_signal(mod_type, snr)
    noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        denoised = model(noisy_tensor).cpu().squeeze().numpy()
    # SNR
    snr_noisy = compute_snr(clean_signal, noisy_signal)
    snr_denoised = compute_snr(clean_signal, denoised)
    print(f"\nSNR зашумленного: {snr_noisy:.2f} дБ | SNR после очистки: {snr_denoised:.2f} дБ")
    # Графики
    plot_signals(message, clean_signal, noisy_signal, denoised, denoised)
    plot_spectra(message, clean_signal, noisy_signal, denoised, denoised)

def main():
    print("1) Проверить денойзер на зашумленном сигнале")
    print("2) Переобучить модель")
    choice = input("Выберите режим (1/2): ").strip()
    if choice == '1':
        interactive_test()
        return
    # Параметры
    DATASET_PATH = "data/denoiser_transformer.npz"
    WEIGHTS_PATH = "weights/transformer_denoiser.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 20
    print("Модель: Transformer Denoiser")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
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
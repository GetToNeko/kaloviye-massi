import numpy as np
import multiprocessing as mp
from pathlib import Path
import json
from tqdm import tqdm

# Константы
SAMPLE_LENGTH = 512
SAMPLING_RATE = 1000  # Hz
SNR_RANGE = (-10, 20)  # dB
NYQUIST_FREQ = SAMPLING_RATE / 2

# Типы модуляции и их метки
LABEL_MAP = {
    'AM': 0,
    'FM': 1,
    'PM': 2,
    'ASK': 3,
    'FSK': 4,
    'BPSK': 5,
    'QPSK': 6
}

class NoiseGenerator:
    @staticmethod
    def awgn(signal, snr_db):
        """Аддитивный белый гауссов шум"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    @staticmethod
    def impulse_noise(signal, prob=0.01, amplitude=5.0):
        """Импульсный шум"""
        mask = np.random.random(signal.shape) < prob
        noise = np.random.normal(0, amplitude, signal.shape) * mask
        return signal + noise
    
    @staticmethod
    def multiplicative_noise(signal, intensity=0.1):
        """Мультипликативный шум"""
        noise = 1 + intensity * np.random.normal(0, 1, signal.shape)
        return signal * noise
    
    @staticmethod
    def frequency_jitter(signal, t, max_drift=0.01):
        """Частотное дрожание"""
        drift = max_drift * np.cumsum(np.random.normal(0, 0.1, len(t)))
        drift = drift - np.mean(drift)  # центрируем дрейф
        return signal * np.exp(1j * 2 * np.pi * drift)

class SignalGenerator:
    @staticmethod
    def generate_message(t, mode='random'):
        """Генерация модулирующего сигнала"""
        if mode == 'sin':
            return np.sin(2 * np.pi * 100 * t)
        elif mode == 'random_sin':
            n_components = np.random.randint(1, 4)
            freqs = np.random.uniform(10, NYQUIST_FREQ/10, size=n_components)
            phases = np.random.uniform(0, 2*np.pi, size=n_components)
            amps = np.random.uniform(0.5, 1.0, size=n_components)
            return sum(a * np.sin(2 * np.pi * f * t + p) 
                      for f, p, a in zip(freqs, phases, amps))
        elif mode == 'nrz':
            # Non-Return-to-Zero кодирование
            symbol_rate = 10  # символов в сигнале
            symbols = np.random.choice([-1, 1], size=symbol_rate)
            return np.repeat(symbols, len(t) // symbol_rate)
        else:
            raise ValueError(f"Unknown message mode: {mode}")
    
    @staticmethod
    def generate_carrier(t, freq=None):
        """Генерация несущего сигнала"""
        if freq is None:
            freq = np.random.uniform(NYQUIST_FREQ/10, NYQUIST_FREQ/3)
        return np.exp(1j * 2 * np.pi * freq * t)

def generate_modulation(t, mod_type, message_mode='random_sin'):
    """Генерация модулированного сигнала
    
    Args:
        t: временная ось
        mod_type: тип модуляции (AM, FM, PM, ASK, FSK, BPSK, QPSK)
        message_mode: режим генерации сообщения:
            - 'sin': простая синусоида
            - 'random_sin': случайная комбинация синусоид
            - 'nrz': Non-Return-to-Zero кодирование
    """
    message = SignalGenerator.generate_message(t, mode=message_mode)
    carrier_freq = np.random.uniform(NYQUIST_FREQ/10, NYQUIST_FREQ/3)
    carrier = SignalGenerator.generate_carrier(t, carrier_freq)
    
    if mod_type == 'AM':
        modulation_index = np.random.uniform(0.3, 0.9)
        signal = (1 + modulation_index * message) * carrier
    
    elif mod_type == 'FM':
        modulation_index = np.random.uniform(0.5, 2.0)
        phase = np.cumsum(message) * modulation_index
        signal = carrier * np.exp(1j * phase)
    
    elif mod_type == 'PM':
        modulation_index = np.random.uniform(0.5, 2.0)
        signal = carrier * np.exp(1j * modulation_index * message)
    
    elif mod_type == 'ASK':
        modulation_index = np.random.uniform(0.3, 0.9)
        signal = (1 + modulation_index * message) * carrier
    
    elif mod_type == 'FSK':
        deviation = np.random.uniform(10, 50)
        signal = np.exp(1j * 2 * np.pi * (carrier_freq + deviation * message) * t)
    
    elif mod_type == 'BPSK':
        signal = carrier * np.sign(message)
    
    elif mod_type == 'QPSK':
        message2 = SignalGenerator.generate_message(t, mode=message_mode)
        signal = carrier * (np.sign(message) + 1j * np.sign(message2)) / np.sqrt(2)
    
    else:
        raise ValueError(f"Unknown modulation type: {mod_type}")
    
    return signal

def add_noise(signal, snr_db):
    """Добавление комбинации шумов"""
    noisy = NoiseGenerator.awgn(signal, snr_db)
    
    # Случайно добавляем другие типы шума
    if np.random.random() < 0.3:  # 30% шанс импульсного шума
        noisy = NoiseGenerator.impulse_noise(noisy)
    
    if np.random.random() < 0.3:  # 30% шанс мультипликативного шума
        noisy = NoiseGenerator.multiplicative_noise(noisy)
    
    if np.random.random() < 0.2:  # 20% шанс частотного дрожания
        t = np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)
        noisy = NoiseGenerator.frequency_jitter(noisy, t)
    
    return noisy

def generate_batch(args):
    """Генерация батча сигналов"""
    mod_type, n_samples, t = args
    
    x_clean = np.zeros((n_samples, SAMPLE_LENGTH), dtype=np.complex64)
    x_noisy = np.zeros((n_samples, SAMPLE_LENGTH), dtype=np.complex64)
    params = []
    
    for i in range(n_samples):
        # Генерация чистого сигнала
        signal = generate_modulation(t, mod_type)
        x_clean[i] = signal
        
        # Добавление шума
        snr = np.random.uniform(*SNR_RANGE)
        x_noisy[i] = add_noise(signal, snr)
        
        # Сохранение параметров
        params.append({
            'snr': float(snr),  # преобразуем в обычное число для JSON
            'mod_type': mod_type,
            'label': LABEL_MAP[mod_type]
        })
    
    return x_clean, x_noisy, params

def generate_dataset(n_samples_per_class, save_path, batch_size=1000):
    """Генерация полного датасета"""
    save_path = Path(save_path)
    save_dir = save_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Подготовка временной оси
    t = np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)
    
    # Подготовка аргументов для multiprocessing
    mod_types = list(LABEL_MAP.keys())
    n_classes = len(mod_types)
    total_samples = n_samples_per_class * n_classes
    
    print(f"Генерация датасета:")
    print(f"Всего классов: {n_classes}")
    print(f"Сэмплов на класс: {n_samples_per_class}")
    print(f"Всего сэмплов: {total_samples}")
    
    # Подготовка массивов для данных
    x_clean = np.zeros((total_samples, 2, SAMPLE_LENGTH), dtype=np.float32)
    x_noisy = np.zeros((total_samples, 2, SAMPLE_LENGTH), dtype=np.float32)
    labels = np.zeros(total_samples, dtype=np.int32)
    all_params = []
    
    # Генерация данных по батчам для каждого класса
    current_idx = 0
    for mod_type in tqdm(mod_types, desc="Классы"):
        remaining_samples = n_samples_per_class
        
        while remaining_samples > 0:
            batch = min(batch_size, remaining_samples)
            x_clean_batch, x_noisy_batch, params = generate_batch((mod_type, batch, t))
            
            # Разделяем комплексные числа на реальную и мнимую части
            x_clean[current_idx:current_idx+batch, 0] = np.real(x_clean_batch)
            x_clean[current_idx:current_idx+batch, 1] = np.imag(x_clean_batch)
            x_noisy[current_idx:current_idx+batch, 0] = np.real(x_noisy_batch)
            x_noisy[current_idx:current_idx+batch, 1] = np.imag(x_noisy_batch)
            labels[current_idx:current_idx+batch] = LABEL_MAP[mod_type]
            
            all_params.extend(params)
            
            # Сохраняем промежуточные результаты каждые 5000 сэмплов
            if current_idx > 0 and current_idx % 5000 == 0:
                temp_save_path = save_dir / f'temp_dataset_{current_idx}.npz'
                np.savez_compressed(
                    temp_save_path,
                    x_clean=x_clean[:current_idx],
                    x_noisy=x_noisy[:current_idx],
                    labels=labels[:current_idx]
                )
                # Сохраняем параметры
                temp_params_path = save_dir / f'temp_params_{current_idx}.json'
                with open(temp_params_path, 'w') as f:
                    json.dump(all_params[:current_idx], f)
            
            current_idx += batch
            remaining_samples -= batch
    
    # Сохраняем финальный датасет
    metadata = {
        'sample_length': SAMPLE_LENGTH,
        'sampling_rate': SAMPLING_RATE,
        'snr_range': SNR_RANGE,
        'label_map': LABEL_MAP
    }
    
    np.savez_compressed(
        save_path,
        x_clean=x_clean,
        x_noisy=x_noisy,
        labels=labels,
        metadata=json.dumps(metadata)
    )
    
    # Сохраняем все параметры в отдельный файл
    params_path = save_dir / 'signal_params.json'
    with open(params_path, 'w') as f:
        json.dump(all_params, f)
    
    print(f"\nДатасет сохранен в {save_path}")
    print(f"Параметры сохранены в {params_path}")
    
    # Удаляем временные файлы
    for temp_file in save_dir.glob('temp_*'):
        temp_file.unlink()

def main():
    # Параметры генерации для двух датасетов
    cnn_samples_per_class = 4285  # ~30000 семплов всего (4285 * 7 = 29995)
    transformer_samples_per_class = 21429  # ~150000 семплов всего (21429 * 7 = 150003)
    
    # Пути сохранения
    save_dir = Path('datasets')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cnn_save_path = save_dir / 'cnn_dataset.npz'
    transformer_save_path = save_dir / 'transformer_dataset.npz'
    
    print("Генерация датасета для CNN...")
    generate_dataset(cnn_samples_per_class, cnn_save_path)
    
    print("\nГенерация датасета для Transformer...")
    generate_dataset(transformer_samples_per_class, transformer_save_path)
    
    # Проверка сгенерированных данных
    print("\n=== Датасет для CNN ===")
    cnn_data = np.load(cnn_save_path)
    print("Размерности:")
    for key in ['x_clean', 'x_noisy', 'labels']:
        print(f"{key}: {cnn_data[key].shape}")
    
    print("\n=== Датасет для Transformer ===")
    transformer_data = np.load(transformer_save_path)
    print("Размерности:")
    for key in ['x_clean', 'x_noisy', 'labels']:
        print(f"{key}: {transformer_data[key].shape}")
    
    # Проверяем распределение классов
    print("\nРаспределение классов:")
    print("CNN датасет:")
    unique, counts = np.unique(cnn_data['labels'], return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Класс {label}: {count} семплов")
    
    print("\nTransformer датасет:")
    unique, counts = np.unique(transformer_data['labels'], return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Класс {label}: {count} семплов")

if __name__ == "__main__":
    main() 

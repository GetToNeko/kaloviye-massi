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
DIGITAL_BITS = 256  # Количество бит для цифровых модуляций

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
        return signal * np.cos(2 * np.pi * drift)  # Теперь применяем к реальному сигналу

class SignalGenerator:
    @staticmethod
    def generate_analog_message(t, mode='random'):
        """Генерация аналогового модулирующего сигнала"""
        if mode == 'sin':
            return np.sin(2 * np.pi * 100 * t)
        elif mode == 'random_sin':
            n_components = np.random.randint(1, 4)
            freqs = np.random.uniform(10, NYQUIST_FREQ/10, size=n_components)
            phases = np.random.uniform(0, 2*np.pi, size=n_components)
            amps = np.random.uniform(0.5, 1.0, size=n_components)
            return sum(a * np.sin(2 * np.pi * f * t + p) 
                      for f, p, a in zip(freqs, phases, amps))
        else:
            raise ValueError(f"Unknown message mode: {mode}")
    
    @staticmethod
    def generate_digital_message(length=DIGITAL_BITS):
        """Генерация цифрового сообщения"""
        return np.random.choice([-1, 1], size=length)
    
    @staticmethod
    def generate_carrier(t, freq=None):
        """Генерация несущего сигнала"""
        if freq is None:
            freq = np.random.uniform(NYQUIST_FREQ/10, NYQUIST_FREQ/3)
        return np.cos(2 * np.pi * freq * t)  # Теперь возвращаем реальный сигнал

def generate_modulation(t, mod_type):
    """Генерация модулированного сигнала
    
    Args:
        t: временная ось
        mod_type: тип модуляции (AM, FM, PM, ASK, FSK, BPSK, QPSK)
    """
    carrier_freq = np.random.uniform(NYQUIST_FREQ/10, NYQUIST_FREQ/3)
    carrier = SignalGenerator.generate_carrier(t, carrier_freq)
    
    if mod_type in ['AM', 'FM', 'PM']:  # Аналоговые модуляции
        message = SignalGenerator.generate_analog_message(t, mode='random_sin')
        
        if mod_type == 'AM':
            modulation_index = np.random.uniform(0.3, 0.9)
            signal = (1 + modulation_index * message) * carrier
        
        elif mod_type == 'FM':
            modulation_index = np.random.uniform(0.5, 2.0)
            phase = np.cumsum(message) * modulation_index
            signal = np.cos(2 * np.pi * carrier_freq * t + phase)
        
        elif mod_type == 'PM':
            modulation_index = np.random.uniform(0.5, 2.0)
            signal = np.cos(2 * np.pi * carrier_freq * t + modulation_index * message)
    
    else:  # Цифровые модуляции
        bits = SignalGenerator.generate_digital_message()
        # Интерполируем биты до нужной длины сигнала
        message = np.repeat(bits, len(t) // len(bits))
        
        if mod_type == 'ASK':
            modulation_index = np.random.uniform(0.3, 0.9)
            signal = (1 + modulation_index * message) * carrier
        
        elif mod_type == 'FSK':
            deviation = np.random.uniform(10, 50)
            signal = np.cos(2 * np.pi * (carrier_freq + deviation * message) * t)
        
        elif mod_type == 'BPSK':
            signal = message * carrier
        
        elif mod_type == 'QPSK':
            # Для QPSK генерируем два потока битов и модулируем их со сдвигом 90°
            bits2 = SignalGenerator.generate_digital_message()
            message2 = np.repeat(bits2, len(t) // len(bits2))
            signal = (message * carrier + message2 * np.cos(2 * np.pi * carrier_freq * t + np.pi/2)) / np.sqrt(2)
    
    return signal

def add_noise(signal, snr_db):
    """Добавление комбинации шумов с контролируемым SNR"""
    # Базовый AWGN с заданным SNR
    noisy = NoiseGenerator.awgn(signal, snr_db)
    
    # Случайно добавляем другие типы шума с меньшей интенсивностью
    if np.random.random() < 0.3:  # 30% шанс импульсного шума
        noisy = NoiseGenerator.impulse_noise(noisy, prob=0.005, amplitude=2.0)
    
    if np.random.random() < 0.3:  # 30% шанс мультипликативного шума
        noisy = NoiseGenerator.multiplicative_noise(noisy, intensity=0.05)
    
    if np.random.random() < 0.2:  # 20% шанс частотного дрожания
        t = np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)
        noisy = NoiseGenerator.frequency_jitter(noisy, t, max_drift=0.005)
    
    # Контроль уровня шума после всех искажений
    signal_power = np.mean(np.abs(signal)**2)
    noise = noisy - signal
    noise_power = np.mean(np.abs(noise)**2)
    current_snr = 10 * np.log10(signal_power / noise_power)
    
    # Корректировка уровня шума для достижения целевого SNR
    if current_snr != snr_db:
        scale = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
        noisy = signal + scale * noise
    
    return noisy

def generate_batch(args):
    """Генерация батча сигналов"""
    mod_type, n_samples, t = args
    
    x_clean = np.zeros((n_samples, SAMPLE_LENGTH), dtype=np.float32)
    x_noisy = np.zeros((n_samples, SAMPLE_LENGTH), dtype=np.float32)
    params = []
    
    for i in range(n_samples):
        # Генерация чистого сигнала
        signal = generate_modulation(t, mod_type)
        x_clean[i] = signal
        
        # Добавление шума с контролируемым SNR
        snr = np.random.uniform(*SNR_RANGE)
        noisy_signal = add_noise(signal, snr)
        x_noisy[i] = noisy_signal
        
        # Сохранение параметров
        params.append({
            'snr': float(snr),
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
    x_clean = np.zeros((total_samples, SAMPLE_LENGTH), dtype=np.float32)
    x_noisy = np.zeros((total_samples, SAMPLE_LENGTH), dtype=np.float32)
    labels = np.zeros(total_samples, dtype=np.int32)
    all_params = []
    
    # Генерация данных по батчам для каждого класса
    current_idx = 0
    for mod_type in tqdm(mod_types, desc="Классы"):
        remaining_samples = n_samples_per_class
        label = LABEL_MAP[mod_type]
        
        while remaining_samples > 0:
            batch_size = min(batch_size, remaining_samples)
            x_clean_batch, x_noisy_batch, params_batch = generate_batch((mod_type, batch_size, t))
            
            # Сохранение результатов
            end_idx = current_idx + batch_size
            x_clean[current_idx:end_idx] = x_clean_batch
            x_noisy[current_idx:end_idx] = x_noisy_batch
            labels[current_idx:end_idx] = label
            all_params.extend(params_batch)
            
            current_idx = end_idx
            remaining_samples -= batch_size
    
    # Нормализация данных
    x_clean = (x_clean - x_clean.mean(axis=1, keepdims=True)) / (x_clean.std(axis=1, keepdims=True) + 1e-8)
    x_noisy = (x_noisy - x_noisy.mean(axis=1, keepdims=True)) / (x_noisy.std(axis=1, keepdims=True) + 1e-8)
    
    # Добавление размерности канала для совместимости с PyTorch
    x_clean = np.expand_dims(x_clean, axis=1)  # Shape: (samples, 1, sequence_length)
    x_noisy = np.expand_dims(x_noisy, axis=1)  # Shape: (samples, 1, sequence_length)
    
    # Сохранение датасета
    np.savez_compressed(
        save_path,
        x_clean=x_clean,
        x_noisy=x_noisy,
        labels=labels
    )
    
    # Сохранение параметров в отдельный JSON файл
    params_path = save_path.with_suffix('.json')
    with open(params_path, 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"\nДатасет сохранен в {save_path}")
    print(f"Параметры сохранены в {params_path}")
    print(f"Форма данных: x_clean: {x_clean.shape}, x_noisy: {x_noisy.shape}, labels: {labels.shape}")

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
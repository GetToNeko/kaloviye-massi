import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Константы
SAMPLE_LENGTH = 512
SAMPLING_RATE = 100000  # 100kHz
BIT_LENGTH = 64
CARRIER_FREQ_RANGE = (1000, 10000)  # 1kHz to 10kHz
SNR_RANGE = (-5, 20)  # dB

# Маппинг меток классов
LABEL_MAP = {
    'FM': 0,
    'AM': 1,
    'PM': 2,
    'FSK': 3,
    'ASK': 4,
    'BPSK': 5,
    'QPSK': 6
}

def generate_time_base():
    """Генерация временной базы для сигналов"""
    return np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)

def generate_carrier(freq):
    """Генерация несущего сигнала"""
    t = generate_time_base()
    return np.cos(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)

def generate_random_bits():
    """Генерация случайной битовой последовательности"""
    return np.random.randint(0, 2, BIT_LENGTH)

def add_noise(signal_i, signal_q, snr_db):
    """Добавление комплексного гауссовского шума с заданным SNR"""
    # Вычисление мощности сигнала
    signal_power = np.mean(signal_i**2 + signal_q**2)
    
    # Вычисление мощности шума на основе SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Генерация комплексного шума
    noise_i = np.random.normal(0, np.sqrt(noise_power/2), len(signal_i))
    noise_q = np.random.normal(0, np.sqrt(noise_power/2), len(signal_q))
    
    return signal_i + noise_i, signal_q + noise_q

# Функции генерации различных типов модуляции
def generate_fm(carrier_freq):
    """Генерация FM сигнала"""
    t = generate_time_base()
    message = np.sin(2 * np.pi * 100 * t)  # Модулирующий сигнал
    beta = 0.5  # Индекс модуляции
    
    phase = 2 * np.pi * carrier_freq * t + beta * np.cumsum(message) * (t[1] - t[0])
    i_signal = np.cos(phase)
    q_signal = np.sin(phase)
    
    return i_signal, q_signal

def generate_am(carrier_freq):
    """Генерация AM сигнала"""
    t = generate_time_base()
    message = np.sin(2 * np.pi * 100 * t)  # Модулирующий сигнал
    m = 0.5  # Глубина модуляции
    
    i_carrier, q_carrier = generate_carrier(carrier_freq)
    i_signal = (1 + m * message) * i_carrier
    q_signal = (1 + m * message) * q_carrier
    
    return i_signal, q_signal

def generate_pm(carrier_freq):
    """Генерация PM сигнала"""
    t = generate_time_base()
    message = np.sin(2 * np.pi * 100 * t)  # Модулирующий сигнал
    beta = 0.5  # Индекс модуляции
    
    phase = 2 * np.pi * carrier_freq * t + beta * message
    i_signal = np.cos(phase)
    q_signal = np.sin(phase)
    
    return i_signal, q_signal

def generate_fsk(carrier_freq):
    """Генерация FSK сигнала"""
    t = generate_time_base()
    bits = generate_random_bits()
    delta_f = 500  # Частотный сдвиг
    
    i_signal = np.zeros(SAMPLE_LENGTH)
    q_signal = np.zeros(SAMPLE_LENGTH)
    
    samples_per_bit = SAMPLE_LENGTH // BIT_LENGTH
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        freq = carrier_freq + (delta_f if bit else -delta_f)
        t_bit = t[start:end]
        i_signal[start:end] = np.cos(2 * np.pi * freq * t_bit)
        q_signal[start:end] = np.sin(2 * np.pi * freq * t_bit)
    
    return i_signal, q_signal

def generate_ask(carrier_freq):
    """Генерация ASK сигнала"""
    t = generate_time_base()
    bits = generate_random_bits()
    
    i_signal = np.zeros(SAMPLE_LENGTH)
    q_signal = np.zeros(SAMPLE_LENGTH)
    
    samples_per_bit = SAMPLE_LENGTH // BIT_LENGTH
    i_carrier, q_carrier = generate_carrier(carrier_freq)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        amplitude = 1.0 if bit else 0.2
        i_signal[start:end] = amplitude * i_carrier[start:end]
        q_signal[start:end] = amplitude * q_carrier[start:end]
    
    return i_signal, q_signal

def generate_bpsk(carrier_freq):
    """Генерация BPSK сигнала"""
    t = generate_time_base()
    bits = generate_random_bits()
    
    i_signal = np.zeros(SAMPLE_LENGTH)
    q_signal = np.zeros(SAMPLE_LENGTH)
    
    samples_per_bit = SAMPLE_LENGTH // BIT_LENGTH
    i_carrier, q_carrier = generate_carrier(carrier_freq)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        phase = 0 if bit else np.pi
        i_signal[start:end] = np.cos(phase) * i_carrier[start:end]
        q_signal[start:end] = np.cos(phase) * q_carrier[start:end]
    
    return i_signal, q_signal

def generate_qpsk(carrier_freq):
    """Генерация QPSK сигнала"""
    t = generate_time_base()
    bits = generate_random_bits()
    
    i_signal = np.zeros(SAMPLE_LENGTH)
    q_signal = np.zeros(SAMPLE_LENGTH)
    
    samples_per_symbol = SAMPLE_LENGTH // (BIT_LENGTH // 2)  # 2 бита на символ
    i_carrier, q_carrier = generate_carrier(carrier_freq)
    
    for i in range(0, BIT_LENGTH, 2):
        symbol_idx = i // 2
        start = symbol_idx * samples_per_symbol
        end = (symbol_idx + 1) * samples_per_symbol
        
        # Определение фазы на основе пары битов
        bit_pair = (bits[i], bits[i+1])
        if bit_pair == (0, 0):
            phase = np.pi/4
        elif bit_pair == (0, 1):
            phase = 3*np.pi/4
        elif bit_pair == (1, 0):
            phase = -np.pi/4
        else:  # (1, 1)
            phase = -3*np.pi/4
            
        i_signal[start:end] = np.cos(phase) * i_carrier[start:end]
        q_signal[start:end] = np.sin(phase) * q_carrier[start:end]
    
    return i_signal, q_signal

def generate_modulation(mod_type):
    """Генерация сигнала заданного типа модуляции"""
    carrier_freq = np.random.uniform(*CARRIER_FREQ_RANGE)
    
    modulation_functions = {
        'FM': generate_fm,
        'AM': generate_am,
        'PM': generate_pm,
        'FSK': generate_fsk,
        'ASK': generate_ask,
        'BPSK': generate_bpsk,
        'QPSK': generate_qpsk
    }
    
    return modulation_functions[mod_type](carrier_freq)

def generate_dataset(n_samples, save_to):
    """Генерация датасета с заданным количеством примеров"""
    x_clean = np.zeros((n_samples, 2, SAMPLE_LENGTH))
    x_noisy = np.zeros((n_samples, 2, SAMPLE_LENGTH))
    labels = np.zeros(n_samples, dtype=int)
    
    mod_types = list(LABEL_MAP.keys())
    
    for i in tqdm(range(n_samples), desc="Генерация датасета"):
        # Выбор случайного типа модуляции
        mod_type = np.random.choice(mod_types)
        
        # Генерация чистого сигнала
        i_signal, q_signal = generate_modulation(mod_type)
        x_clean[i, 0] = i_signal
        x_clean[i, 1] = q_signal
        
        # Добавление шума
        snr = np.random.uniform(*SNR_RANGE)
        i_noisy, q_noisy = add_noise(i_signal, q_signal, snr)
        x_noisy[i, 0] = i_noisy
        x_noisy[i, 1] = q_noisy
        
        # Сохранение метки класса
        labels[i] = LABEL_MAP[mod_type]
    
    # Сохранение датасета
    np.savez(save_to,
             x_clean=x_clean,
             x_noisy=x_noisy,
             labels=labels)
    
    return x_clean, x_noisy, labels

def validate_dataset(x_clean, x_noisy, labels):
    """Простая валидация сгенерированного датасета"""
    print(f"Размер датасета:")
    print(f"x_clean: {x_clean.shape}")
    print(f"x_noisy: {x_noisy.shape}")
    print(f"labels: {labels.shape}")
    
    print("\nРаспределение классов:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        mod_type = [k for k, v in LABEL_MAP.items() if v == label][0]
        print(f"{mod_type}: {count}")

def plot_example(x_clean, x_noisy, label, index):
    """Визуализация примера из датасета"""
    mod_type = [k for k, v in LABEL_MAP.items() if v == label][0]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.plot(x_clean[index, 0], label='I')
    plt.plot(x_clean[index, 1], label='Q')
    plt.title(f'Чистый сигнал ({mod_type})')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(x_noisy[index, 0], label='I')
    plt.plot(x_noisy[index, 1], label='Q')
    plt.title('Зашумленный сигнал')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Генерация полного датасета
    TOTAL_SAMPLES = 150000  # Общее количество примеров
    CNN_SAMPLES = 20000    # Количество примеров для CNN
    
    # Генерация датасета
    print("Генерация полного датасета...")
    x_clean, x_noisy, labels = generate_dataset(TOTAL_SAMPLES, "full_dataset.npz")
    
    # Валидация
    validate_dataset(x_clean, x_noisy, labels)
    
    # Создание подмножества для CNN
    print("\nСоздание подмножества для CNN...")
    indices = np.random.choice(TOTAL_SAMPLES, CNN_SAMPLES, replace=False)
    np.savez("cnn_dataset.npz",
             x_clean=x_clean[indices],
             x_noisy=x_noisy[indices],
             labels=labels[indices])
    
    # Визуализация примера
    plot_example(x_clean, x_noisy, labels[0], 0) 

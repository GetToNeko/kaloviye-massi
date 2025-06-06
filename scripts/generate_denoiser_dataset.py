import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import numpy as np
from scripts.generate_dataset_improved import SignalGenerator, add_noise, LABEL_MAP, SAMPLE_LENGTH, SAMPLING_RATE

# Настройки по умолчанию
CNN_SEQ_LEN = 512
TRANSFORMER_SEQ_LEN = 512
DEFAULT_MOD_TYPES = list(LABEL_MAP.keys())
DEFAULT_SNR_RANGE = (0, 20)


def generate_denoiser_dataset(n_samples, seq_len, mod_types, snr_range, save_path):
    print(f"Генерация датасета для денойзера:")
    print(f"  Примеров: {n_samples}")
    print(f"  Длина сигнала: {seq_len}")
    print(f"  Типы модуляции: {mod_types}")
    print(f"  SNR: {snr_range}")
    print(f"  Сохранение в: {save_path}")

    x_clean = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    x_noisy = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)

    t = np.linspace(0, seq_len / SAMPLING_RATE, seq_len)

    for i in range(n_samples):
        mod_type = np.random.choice(mod_types)
        label = LABEL_MAP[mod_type]
        # Генерация чистого сигнала
        if mod_type in ['AM', 'FM', 'PM']:
            message = SignalGenerator.generate_analog_message(t, mode='random_sin')
            carrier_freq = np.random.uniform(SAMPLING_RATE/20, SAMPLING_RATE/2/3)
            carrier = SignalGenerator.generate_carrier(t, carrier_freq)
            if mod_type == 'AM':
                modulation_index = np.random.uniform(0.3, 0.9)
                clean_signal = (1 + modulation_index * message) * carrier
            elif mod_type == 'FM':
                modulation_index = np.random.uniform(0.5, 2.0)
                phase = np.cumsum(message) * modulation_index
                clean_signal = np.cos(2 * np.pi * carrier_freq * t + phase)
            elif mod_type == 'PM':
                modulation_index = np.random.uniform(0.5, 2.0)
                clean_signal = np.cos(2 * np.pi * carrier_freq * t + modulation_index * message)
        else:
            bits = SignalGenerator.generate_digital_message()
            message = np.repeat(bits, seq_len // len(bits))
            carrier_freq = np.random.uniform(SAMPLING_RATE/20, SAMPLING_RATE/2/3)
            carrier = SignalGenerator.generate_carrier(t, carrier_freq)
            if mod_type == 'ASK':
                modulation_index = np.random.uniform(0.3, 0.9)
                clean_signal = (1 + modulation_index * message) * carrier
            elif mod_type == 'FSK':
                deviation = np.random.uniform(10, 50)
                clean_signal = np.cos(2 * np.pi * (carrier_freq + deviation * message) * t)
            elif mod_type == 'BPSK':
                clean_signal = message * carrier
            elif mod_type == 'QPSK':
                bits2 = SignalGenerator.generate_digital_message()
                message2 = np.repeat(bits2, seq_len // len(bits2))
                clean_signal = (message * carrier + message2 * np.cos(2 * np.pi * carrier_freq * t + np.pi/2)) / np.sqrt(2)
        # Добавление шума
        snr = np.random.uniform(*snr_range)
        noisy_signal = add_noise(clean_signal, snr)
        # Нормализация
        clean_signal = (clean_signal - clean_signal.mean()) / (clean_signal.std() + 1e-8)
        noisy_signal = (noisy_signal - noisy_signal.mean()) / (noisy_signal.std() + 1e-8)
        # Сохраняем
        x_clean[i, 0, :] = clean_signal
        x_noisy[i, 0, :] = noisy_signal
        labels[i] = label
        if (i+1) % 1000 == 0 or i == n_samples-1:
            print(f"  {i+1}/{n_samples} сэмплов готово...")
    np.savez_compressed(save_path, x_clean=x_clean, x_noisy=x_noisy, labels=labels)
    print(f"Датасет сохранён: {save_path}")


def main():
    print("\nГенератор датасетов для денойзеров (CNN/Transformer)")
    print("1) Для CNN (~512 отсчётов, 20 000 примеров)")
    print("2) Для Transformer (~512 отсчёта, 150 000 примеров)")
    choice = input("Выберите режим (1/2): ").strip()
    if choice == '1':
        n_samples = 20000
        seq_len = CNN_SEQ_LEN
        save_path = 'data/denoiser_cnn.npz'
    elif choice == '2':
        n_samples = 150000
        seq_len = TRANSFORMER_SEQ_LEN
        save_path = 'data/denoiser_transformer.npz'
    else:
        print("Неверный выбор!")
        return
    mod_types = DEFAULT_MOD_TYPES
    snr_range = DEFAULT_SNR_RANGE
    generate_denoiser_dataset(n_samples, seq_len, mod_types, snr_range, save_path)

if __name__ == "__main__":
    main() 
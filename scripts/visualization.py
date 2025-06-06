import numpy as np
import matplotlib.pyplot as plt
from scripts.generate_dataset_improved import SAMPLE_LENGTH, SAMPLING_RATE

def plot_signals(message_signal, clean_signal, noisy_signal, cnn_denoised, transformer_denoised, save_path=None):
    """Отображение сигналов (одноканальные)"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Сравнение CNN и Transformer', fontsize=16)

    # Первая строка: исходный модулирующий сигнал
    axes[0, 0].plot(message_signal, color='purple')
    axes[0, 0].set_ylabel('Модулирующий')
    axes[0, 0].grid(True)
    axes[0, 1].plot(message_signal, color='purple')
    axes[0, 1].set_ylabel('Модулирующий')
    axes[0, 1].grid(True)

    # Вторая строка: чистый модулированный сигнал
    axes[1, 0].plot(clean_signal, color='b')
    axes[1, 0].set_ylabel('Модулированный')
    axes[1, 0].grid(True)
    axes[1, 1].plot(clean_signal, color='b')
    axes[1, 1].set_ylabel('Модулированный')
    axes[1, 1].grid(True)

    # Третья строка: зашумленный сигнал
    axes[2, 0].plot(noisy_signal, color='orange')
    axes[2, 0].set_ylabel('Зашумленный')
    axes[2, 0].grid(True)
    axes[2, 1].plot(noisy_signal, color='orange')
    axes[2, 1].set_ylabel('Зашумленный')
    axes[2, 1].grid(True)

    # Четвертая строка: очищенные сигналы
    axes[3, 0].plot(cnn_denoised, color='g')
    axes[3, 0].set_ylabel('Очищенный')
    axes[3, 0].grid(True)
    axes[3, 1].plot(transformer_denoised, color='g')
    axes[3, 1].set_ylabel('Очищенный')
    axes[3, 1].grid(True)

    for ax in axes.flat:
        ax.set_xlabel('Отсчёт')

    axes[0, 0].set_title('CNN')
    axes[0, 1].set_title('Transformer')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_spectra(message_signal, clean_signal, noisy_signal, cnn_denoised, transformer_denoised, save_path=None):
    """Отображение спектров сигналов: отдельно для CNN и Transformer"""
    freq = np.fft.fftfreq(SAMPLE_LENGTH, 1/SAMPLING_RATE)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('Спектры сигналов', fontsize=16)

    # Спектр модулирующего сигнала
    spectrum = np.abs(np.fft.fft(message_signal))
    axes[0, 0].plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2], color='purple')
    axes[0, 0].set_title('Модулирующий сигнал')
    axes[0, 0].set_xlabel('Частота (Гц)')
    axes[0, 0].set_ylabel('Амплитуда')
    axes[0, 0].grid(True)

    # Спектр модулированного сигнала (чистого)
    spectrum = np.abs(np.fft.fft(clean_signal))
    axes[0, 1].plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2], color='b')
    axes[0, 1].set_title('Модулированный сигнал')
    axes[0, 1].set_xlabel('Частота (Гц)')
    axes[0, 1].set_ylabel('Амплитуда')
    axes[0, 1].grid(True)

    # Спектр зашумленного сигнала
    spectrum = np.abs(np.fft.fft(noisy_signal))
    axes[0, 2].plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2], color='orange')
    axes[0, 2].set_title('Зашумленный сигнал')
    axes[0, 2].set_xlabel('Частота (Гц)')
    axes[0, 2].set_ylabel('Амплитуда')
    axes[0, 2].grid(True)

    # Спектр очищенного сигнала (CNN)
    spectrum = np.abs(np.fft.fft(cnn_denoised))
    axes[1, 0].plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2], color='g')
    axes[1, 0].set_title('Очищенный (CNN)')
    axes[1, 0].set_xlabel('Частота (Гц)')
    axes[1, 0].set_ylabel('Амплитуда')
    axes[1, 0].grid(True)

    # Спектр очищенного сигнала (Transformer)
    spectrum = np.abs(np.fft.fft(transformer_denoised))
    axes[1, 1].plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2], color='lime')
    axes[1, 1].set_title('Очищенный (Transformer)')
    axes[1, 1].set_xlabel('Частота (Гц)')
    axes[1, 1].set_ylabel('Амплитуда')
    axes[1, 1].grid(True)

    # Пустой график для симметрии
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def compute_snr(clean, test):
    """Вычисляет SNR (дБ) между чистым и тестовым сигналом"""
    power_signal = np.mean(clean ** 2)
    power_noise = np.mean((clean - test) ** 2)
    if power_noise == 0:
        return float('inf')
    return 10 * np.log10(power_signal / power_noise) 
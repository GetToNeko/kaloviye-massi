import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def compute_snr(clean, noisy):
    power_signal = np.mean(clean ** 2)
    power_noise = np.mean((clean - noisy) ** 2)
    if power_noise == 0:
        return float('inf')
    return 10 * np.log10(power_signal / power_noise)

def main():
    path = input('Путь к датасету (.npz): ').strip()
    data = np.load(path)
    x_clean = data['x_clean']
    x_noisy = data['x_noisy']
    labels = data['labels'] if 'labels' in data else None
    N = x_clean.shape[0]
    print(f"\nРазмерность x_clean: {x_clean.shape}")
    print(f"Размерность x_noisy: {x_noisy.shape}")
    if labels is not None:
        print(f"Размерность labels: {labels.shape}")
    print(f"\nСтатистика по x_clean:")
    print(f"  min: {x_clean.min():.4f}, max: {x_clean.max():.4f}, mean: {x_clean.mean():.4f}, std: {x_clean.std():.4f}")
    print(f"Статистика по x_noisy:")
    print(f"  min: {x_noisy.min():.4f}, max: {x_noisy.max():.4f}, mean: {x_noisy.mean():.4f}, std: {x_noisy.std():.4f}")
    # SNR по всему датасету
    snrs = np.array([compute_snr(x_clean[i,0], x_noisy[i,0]) for i in range(N)])
    print(f"\nSNR (min/mean/max): {snrs.min():.2f} / {snrs.mean():.2f} / {snrs.max():.2f} дБ")
    # Распределение SNR
    plt.figure(figsize=(6,3))
    plt.hist(snrs, bins=40, color='gray', alpha=0.7)
    plt.title('Распределение SNR по датасету')
    plt.xlabel('SNR, дБ')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.show()
    # Распределение меток
    if labels is not None:
        plt.figure(figsize=(6,3))
        plt.hist(labels, bins=np.arange(labels.min(), labels.max()+2)-0.5, color='blue', alpha=0.7, rwidth=0.8)
        plt.title('Распределение меток модуляции')
        plt.xlabel('Метка')
        plt.ylabel('Частота')
        plt.tight_layout()
        plt.show()
    # Примеры сигналов
    idxs = np.random.choice(N, size=5, replace=False)
    for idx in idxs:
        fig, axes = plt.subplots(2,2, figsize=(10,6))
        axes[0,0].plot(x_clean[idx,0], label='Чистый')
        axes[0,0].plot(x_noisy[idx,0], label='Зашумленный', alpha=0.7)
        axes[0,0].set_title(f'Сигналы (пример {idx})')
        axes[0,0].legend()
        axes[0,0].grid(True)
        axes[0,1].plot(x_noisy[idx,0] - x_clean[idx,0], color='orange')
        axes[0,1].set_title('Шум (разность)')
        axes[0,1].grid(True)
        # Спектры
        spectrum_clean = np.abs(np.fft.fft(x_clean[idx,0]))
        spectrum_noisy = np.abs(np.fft.fft(x_noisy[idx,0]))
        freq = np.fft.fftfreq(x_clean.shape[-1], 1/1000)
        axes[1,0].plot(freq[:len(freq)//2], spectrum_clean[:len(freq)//2], label='Чистый')
        axes[1,0].plot(freq[:len(freq)//2], spectrum_noisy[:len(freq)//2], label='Зашумленный', alpha=0.7)
        axes[1,0].set_title('Спектры')
        axes[1,0].legend()
        axes[1,0].grid(True)
        axes[1,1].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 
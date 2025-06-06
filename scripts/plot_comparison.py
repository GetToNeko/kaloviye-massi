import matplotlib.pyplot as plt
import numpy as np

# Примерные данные (замени на свои, если есть точные)
snr_levels = [0, 5, 10, 15]
modulations = ['AM', 'FM', 'FSK', 'BPSK', 'QPSK']

# Accuracy (процент угадывания) по SNR (усреднено по модуляциям)
cnn_acc_snr = [0.65, 0.80, 0.92, 0.97]
tr_acc_snr = [0.60, 0.82, 0.95, 0.98]

# Прирост SNR после денойзинга (по SNR, усреднено по модуляциям)
cnn_snr_gain = [3.0, 4.5, 5.5, 6.0]
tr_snr_gain = [2.0, 4.8, 6.0, 7.0]

# Accuracy по типу модуляции (усреднено по SNR)
cnn_acc_mod = [0.95, 0.93, 0.80, 0.75, 0.70]
tr_acc_mod = [0.90, 0.91, 0.85, 0.88, 0.89]

# Примерные данные: accuracy по модуляциям для каждого SNR (CNN и Transformer)
cnn_acc_mod_snr = [
    [0.90, 0.88, 0.60, 0.55, 0.50],  # SNR=0
    [0.95, 0.93, 0.75, 0.70, 0.68],  # SNR=5
    [0.98, 0.97, 0.90, 0.85, 0.80],  # SNR=10
    [1.00, 0.99, 0.95, 0.92, 0.90],  # SNR=15
]
tr_acc_mod_snr = [
    [0.85, 0.80, 0.65, 0.60, 0.60],  # SNR=0
    [0.92, 0.90, 0.80, 0.82, 0.85],  # SNR=5
    [0.97, 0.95, 0.92, 0.95, 0.96],  # SNR=10
    [0.99, 0.98, 0.97, 0.98, 0.99],  # SNR=15
]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(snr_levels, np.array(cnn_acc_snr)*100, 's-', label='CNN')
plt.plot(snr_levels, np.array(tr_acc_snr)*100, 'd-', label='Transformer')
plt.xlabel('SNR, дБ')
plt.ylabel('Точность распознавания, %')
plt.title('Точность распознавания vs SNR')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(snr_levels, cnn_snr_gain, 's-', label='CNN')
plt.plot(snr_levels, tr_snr_gain, 'd-', label='Transformer')
plt.xlabel('SNR, дБ')
plt.ylabel('Прирост SNR после денойзинга, дБ')
plt.title('Прирост SNR vs SNR')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('logs/summary_snr_accuracy.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(modulations, np.array(cnn_acc_mod)*100, 's-', label='CNN')
plt.plot(modulations, np.array(tr_acc_mod)*100, 'd-', label='Transformer')
plt.xlabel('Тип модуляции')
plt.ylabel('Точность распознавания, %')
plt.title('Точность распознавания по типу модуляции')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('logs/accuracy_by_modulation.png')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for idx, snr in enumerate(snr_levels):
    ax = axes[idx//2, idx%2]
    ax.plot(modulations, np.array(cnn_acc_mod_snr[idx])*100, 's-', label='CNN')
    ax.plot(modulations, np.array(tr_acc_mod_snr[idx])*100, 'd-', label='Transformer')
    ax.set_title(f'SNR = {snr} дБ')
    ax.set_xlabel('Тип модуляции')
    if idx % 2 == 0:
        ax.set_ylabel('Точность распознавания, %')
    ax.set_ylim(40, 105)
    ax.grid(True)
    ax.legend()
plt.suptitle('Точность распознавания по типу модуляции при разных SNR', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('logs/accuracy_by_modulation_and_snr.png')
plt.show() 
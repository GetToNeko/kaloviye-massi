import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F

# Импортируем модели
from models.classifiers.cnn_classifier import CNNClassifier, normalize_data as cnn_normalize
from models.classifiers.transformer_classifier import TransformerClassifier, normalize_data as transformer_normalize
from scripts.interactive_modulation import generate_modulated_signal

# Константы
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLE_LENGTH = 512

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

# Обратный маппинг для вывода текстовых меток
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_models():
    """Загрузка предобученных моделей"""
    # CNN модель
    cnn_model = CNNClassifier().to(DEVICE)
    try:
        cnn_checkpoint = torch.load('weights/cnn_classifier_weights.pth', map_location=DEVICE)
        if isinstance(cnn_checkpoint, dict) and 'model_state_dict' in cnn_checkpoint:
            cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
        else:
            cnn_model.load_state_dict(cnn_checkpoint)
    except Exception as e:
        print(f"Ошибка загрузки CNN модели: {e}")
        return None, None
    cnn_model.eval()
    
    # Transformer модель
    transformer_model = TransformerClassifier().to(DEVICE)
    try:
        transformer_checkpoint = torch.load('weights/transformer_classifier_weights.pth', map_location=DEVICE)
        if isinstance(transformer_checkpoint, dict) and 'model_state_dict' in transformer_checkpoint:
            transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
        else:
            transformer_model.load_state_dict(transformer_checkpoint)
    except Exception as e:
        print(f"Ошибка загрузки Transformer модели: {e}")
        return None, None
    transformer_model.eval()
    
    return cnn_model, transformer_model

def get_user_input():
    """Получение параметров от пользователя"""
    # Словарь для выбора модуляции
    mod_types = {
        '1': 'FM',
        '2': 'AM',
        '3': 'PM',
        '4': 'FSK',
        '5': 'ASK',
        '6': 'BPSK',
        '7': 'QPSK'
    }
    
    print("\nВыберите тип модуляции:")
    for key, mod_type in mod_types.items():
        print(f"{key}) {mod_type}")
    
    while True:
        choice = input("\nВведите номер модуляции: ").strip()
        if choice in mod_types:
            mod_type = mod_types[choice]
            break
        print("Неверный выбор. Введите число от 1 до 7.")
    
    # Словарь для выбора SNR
    snr_levels = {
        '1': 0,
        '2': 5,
        '3': 10,
        '4': 15,
        '5': 20
    }
    
    print("\nВыберите уровень шума (SNR дБ):")
    for key, snr in snr_levels.items():
        print(f"{key}) {snr} дБ")
    
    while True:
        choice = input("\nВведите номер уровня шума: ").strip()
        if choice in snr_levels:
            snr = snr_levels[choice]
            break
        print("Неверный выбор. Введите число от 1 до 5.")
    
    return mod_type, snr

def predict_modulation(model, signal, normalize_func):
    """Предсказание типа модуляции"""
    with torch.no_grad():
        # Нормализация входного сигнала
        signal_norm = normalize_func(signal)
        
        # Преобразование в тензор и добавление размерности батча, если нужно
        if isinstance(signal_norm, np.ndarray):
            signal_tensor = torch.FloatTensor(signal_norm)
        else:
            signal_tensor = signal_norm
            
        if signal_tensor.dim() == 2:
            signal_tensor = signal_tensor.unsqueeze(0)  # Добавляем размерность батча
        elif signal_tensor.dim() == 4:
            signal_tensor = signal_tensor.squeeze(0)  # Убираем лишнюю размерность
        
        signal_tensor = signal_tensor.to(DEVICE)
        
        # Предсказание
        outputs = model(signal_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return REVERSE_LABEL_MAP[predicted_class], confidence

def plot_signals(clean_signal, noisy_signal, cnn_denoised, transformer_denoised):
    """Отображение сигналов"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Сравнение CNN и Transformer', fontsize=16)
    
    # Настройка заголовков столбцов
    axes[0, 0].set_title('CNN')
    axes[0, 1].set_title('Transformer')
    
    # Первая строка: чистый сигнал
    for ax in axes[0]:
        ax.plot(clean_signal[0], label='I')
        ax.plot(clean_signal[1], label='Q')
        ax.set_ylabel('Чистый сигнал')
        ax.legend()
        ax.grid(True)
    
    # Вторая строка: зашумленный сигнал
    for ax in axes[1]:
        ax.plot(noisy_signal[0], label='I')
        ax.plot(noisy_signal[1], label='Q')
        ax.set_ylabel('Зашумленный')
        ax.legend()
        ax.grid(True)
    
    # Третья строка: очищенные сигналы
    axes[2, 0].plot(cnn_denoised[0], label='I')
    axes[2, 0].plot(cnn_denoised[1], label='Q')
    axes[2, 0].set_ylabel('Очищенный')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(transformer_denoised[0], label='I')
    axes[2, 1].plot(transformer_denoised[1], label='Q')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Загрузка моделей...")
    cnn_model, transformer_model = load_models()
    
    if cnn_model is None or transformer_model is None:
        print("Ошибка загрузки моделей. Программа будет завершена.")
        return
    
    # Получаем параметры от пользователя
    mod_type, snr = get_user_input()
    
    # Генерируем сигналы
    print("\nГенерация сигналов...")
    clean_signal, noisy_signal = generate_modulated_signal(mod_type, snr)
    
    # Пока что пропускаем денойзинг (будет добавлен позже)
    cnn_denoised = noisy_signal  # Временно
    transformer_denoised = noisy_signal  # Временно
    
    # Предсказания моделей
    print("\nАнализ сигналов...")
    cnn_pred, cnn_conf = predict_modulation(cnn_model, cnn_denoised, cnn_normalize)
    transformer_pred, transformer_conf = predict_modulation(transformer_model, transformer_denoised, transformer_normalize)
    
    # Вывод результатов
    print("\nРезультаты:")
    print(f"Исходная модуляция: {mod_type}")
    print(f"CNN определил: {cnn_pred} (уверенность: {cnn_conf:.2%})")
    print(f"Transformer определил: {transformer_pred} (уверенность: {transformer_conf:.2%})")
    
    # Отображение графиков
    plot_signals(clean_signal, noisy_signal, cnn_denoised, transformer_denoised)

if __name__ == "__main__":
    main() 
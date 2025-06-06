import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from datetime import datetime
import os

# Импортируем модели
from models.classifiers.cnn_classifier import CNNClassifier, normalize_data as cnn_normalize
from models.classifiers.transformer_classifier import TransformerClassifier, normalize_data as transformer_normalize
from scripts.interactive_modulation import generate_modulated_signal
from scripts.generate_dataset_improved import SignalGenerator, SAMPLING_RATE
from models.denoisers.cnn_denoiser import CNNDenoiser
from models.denoisers.transformer_denoiser import TransformerDenoiser
from scripts.visualization import plot_signals, plot_spectra, compute_snr

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
        transformer_checkpoint = torch.load('weights/transformer_classifier_weights_v2.pth', map_location=DEVICE)
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
    # Словарь для выбора SNR (как в interactive_modulation.py)
    snr_levels = {
        '1': -10,
        '2': -5,
        '3': 0,
        '4': 5,
        '5': 10,
        '6': 15,
        '7': 20,
        '8': 25,
        '9': 30
    }
    print("\nВыберите уровень шума (SNR дБ):")
    for key, snr in snr_levels.items():
        print(f"{key}) {snr} дБ")
    while True:
        choice = input("\nВведите номер уровня шума: ").strip()
        if choice in snr_levels:
            snr = snr_levels[choice]
            break
        print("Неверный выбор. Введите число от 1 до 9.")
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

def main():
    print("Загрузка моделей...")
    cnn_model, transformer_model = load_models()
    # Загрузка денойзеров
    cnn_denoiser = CNNDenoiser().to(DEVICE)
    transformer_denoiser = TransformerDenoiser().to(DEVICE)
    try:
        cnn_checkpoint = torch.load('weights/cnn_denoiser.pth', map_location=DEVICE)
        if isinstance(cnn_checkpoint, dict) and 'model_state_dict' in cnn_checkpoint:
            cnn_denoiser.load_state_dict(cnn_checkpoint['model_state_dict'])
        else:
            cnn_denoiser.load_state_dict(cnn_checkpoint)
        cnn_denoiser.eval()
    except Exception as e:
        print(f"Ошибка загрузки весов CNN-денойзера: {e}")
        cnn_denoiser = None
    try:
        transformer_checkpoint = torch.load('weights/transformer_denoiser.pth', map_location=DEVICE)
        if isinstance(transformer_checkpoint, dict) and 'model_state_dict' in transformer_checkpoint:
            transformer_denoiser.load_state_dict(transformer_checkpoint['model_state_dict'])
        else:
            transformer_denoiser.load_state_dict(transformer_checkpoint)
        transformer_denoiser.eval()
    except Exception as e:
        print(f"Ошибка загрузки весов Transformer-денойзера: {e}")
        transformer_denoiser = None
    if cnn_model is None or transformer_model is None:
        print("Ошибка загрузки моделей. Программа будет завершена.")
        return
    # Получаем параметры от пользователя
    mod_type, snr = get_user_input()
    # Формируем имя для сохранения графиков
    os.makedirs('logs', exist_ok=True)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'{mod_type}_{snr}dB_{now}'
    signals_path = os.path.join('logs', f'{base_name}_signals.png')
    spectra_path = os.path.join('logs', f'{base_name}_spectra.png')
    # Генерируем сигналы
    print("\nГенерация сигналов...")
    message_signal, clean_signal, noisy_signal = generate_modulated_signal(mod_type, snr)
    # --- Применяем денойзеры ---
    noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, seq_len)
    with torch.no_grad():
        if cnn_denoiser is not None:
            cnn_denoised_tensor = cnn_denoiser(noisy_tensor).cpu().squeeze().numpy()
        else:
            cnn_denoised_tensor = noisy_signal
        if transformer_denoiser is not None:
            transformer_denoised_tensor = transformer_denoiser(noisy_tensor).cpu().squeeze().numpy()
        else:
            transformer_denoised_tensor = noisy_signal
    # --- Отладочный вывод ---
    print("[DEBUG] noisy_signal[:5]           ", noisy_signal[:5])
    print("[DEBUG] cnn_denoised_tensor[:5]    ", cnn_denoised_tensor[:5])
    print("[DEBUG] transformer_denoised_tensor[:5]", transformer_denoised_tensor[:5])
    if np.allclose(noisy_signal, cnn_denoised_tensor):
        print("[ОШИБКА] CNN-денойзер не изменяет сигнал! Проверьте загрузку весов и применение модели.")
    if np.allclose(noisy_signal, transformer_denoised_tensor):
        print("[ОШИБКА] Transformer-денойзер не изменяет сигнал! Проверьте загрузку весов и применение модели.")
    # --- Классификация по очищенным сигналам ---
    print("\nАнализ сигналов...")
    cnn_pred, cnn_conf = predict_modulation(cnn_model, cnn_denoised_tensor, cnn_normalize)
    transformer_pred, transformer_conf = predict_modulation(transformer_model, transformer_denoised_tensor, transformer_normalize)
    # Оценка качества шумоподавления
    snr_noisy = compute_snr(clean_signal, noisy_signal)
    snr_cnn = compute_snr(clean_signal, cnn_denoised_tensor)
    snr_transformer = compute_snr(clean_signal, transformer_denoised_tensor)
    print(f"\nSNR зашумленного сигнала: {snr_noisy:.2f} дБ")
    print(f"SNR после шумоподавления (CNN): {snr_cnn:.2f} дБ")
    print(f"SNR после шумоподавления (Transformer): {snr_transformer:.2f} дБ")
    # Диагностика SNR
    if abs(snr_cnn - snr_noisy) < 1e-2 and abs(snr_transformer - snr_noisy) < 1e-2:
        print("[ВНИМАНИЕ] SNR после шумоподавления почти не изменился! Проверьте работу денойзеров и веса.")
    print(f"[DEBUG] CNN очищенный: min={cnn_denoised_tensor.min():.3f}, max={cnn_denoised_tensor.max():.3f}")
    print(f"[DEBUG] Transformer очищенный: min={transformer_denoised_tensor.min():.3f}, max={transformer_denoised_tensor.max():.3f}")
    # Вывод результатов
    print("\nРезультаты:")
    print(f"Исходная модуляция: {mod_type}")
    print(f"CNN определил: {cnn_pred} (уверенность: {cnn_conf:.2%})")
    print(f"Transformer определил: {transformer_pred} (уверенность: {transformer_conf:.2%})")
    # Отображение и сохранение графиков
    plot_signals(message_signal, clean_signal, noisy_signal, cnn_denoised_tensor, transformer_denoised_tensor, save_path=signals_path)
    plot_spectra(message_signal, clean_signal, noisy_signal, cnn_denoised_tensor, transformer_denoised_tensor, save_path=spectra_path)
    print(f"\nГрафики сохранены: {signals_path} и {spectra_path}")

if __name__ == "__main__":
    main() 
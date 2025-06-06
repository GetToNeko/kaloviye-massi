import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scripts.generate_dataset_improved import (
    generate_modulation, SignalGenerator, add_noise,
    SAMPLE_LENGTH, SAMPLING_RATE
)

# Константы
CARRIER_FREQ_RANGE = (1000, 10000)  # 1kHz to 10kHz
MOD_FREQ_RANGE = (50, 500)  # 50-500 Hz

class ModulationGenerator:
    def __init__(self):
        self.mod_types = {
            '1': ('FM', 'FM'),
            '2': ('AM', 'AM'),
            '3': ('PM', 'PM'),
            '4': ('FSK', 'FSK'),
            '5': ('ASK', 'ASK'),
            '6': ('BPSK', 'BPSK'),
            '7': ('QPSK', 'QPSK')
        }
        
        self.snr_levels = {
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
        
        self.current_mod_signal = None
        self.current_noisy_signal = None
        self.current_message = None
        self.current_mod_type = None
    
    def print_menu(self):
        print("\nВыберите тип модуляции:")
        for key, (display_name, _) in self.mod_types.items():
            print(f"{key}) {display_name}")
        print("q) Выход")
    
    def print_snr_menu(self):
        print("\nВыберите уровень шума (SNR дБ):")
        for key, snr in self.snr_levels.items():
            print(f"{key}) {snr}")
    
    def generate_signal(self, mod_choice, snr_choice):
        """Генерация сигнала с выбранными параметрами"""
        if mod_choice not in self.mod_types:
            return False
        
        # Получение параметров
        display_name, mod_type = self.mod_types[mod_choice]
        snr = self.snr_levels[snr_choice]
        
        # Генерация временной оси
        t = np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)
        
        # Генерация модулированного сигнала (теперь одномерного)
        self.current_mod_signal = generate_modulation(t, mod_type)
        
        # Добавление шума
        self.current_noisy_signal = add_noise(self.current_mod_signal, snr)
        
        # Сохраняем модулирующий сигнал (для отображения)
        if mod_type in ['AM', 'FM', 'PM']:
            self.current_message = SignalGenerator.generate_analog_message(t, mode='random_sin')
        else:
            bits = SignalGenerator.generate_digital_message()
            self.current_message = np.repeat(bits, len(t) // len(bits))
        
        self.current_mod_type = display_name
        return True
    
    def plot_signals(self):
        """Отображение графиков сигналов"""
        if self.current_mod_signal is None:
            print("Сначала сгенерируйте сигнал!")
            return
        
        fig = plt.figure(figsize=(15, 12))
        
        # График модулирующего сигнала
        plt.subplot(321)
        plt.plot(self.current_message)
        plt.title(f'Модулирующий сигнал ({self.current_mod_type})')
        plt.grid(True)
        
        # График чистого модулированного сигнала
        plt.subplot(322)
        plt.plot(self.current_mod_signal)
        plt.title('Модулированный сигнал')
        plt.grid(True)
        
        # График зашумленного сигнала
        plt.subplot(323)
        plt.plot(self.current_noisy_signal)
        plt.title('Зашумленный сигнал')
        plt.grid(True)
        
        # Спектр модулирующего сигнала
        plt.subplot(324)
        freq = np.fft.fftfreq(SAMPLE_LENGTH, 1/SAMPLING_RATE)
        spectrum = np.abs(np.fft.fft(self.current_message))
        plt.plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2])
        plt.title('Спектр модулирующего сигнала')
        plt.xlabel('Частота (Гц)')
        plt.grid(True)
        
        # Спектр чистого сигнала
        plt.subplot(325)
        spectrum = np.abs(np.fft.fft(self.current_mod_signal))
        plt.plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2])
        plt.title('Спектр модулированного сигнала')
        plt.xlabel('Частота (Гц)')
        plt.grid(True)
        
        # Спектр зашумленного сигнала
        plt.subplot(326)
        spectrum = np.abs(np.fft.fft(self.current_noisy_signal))
        plt.plot(freq[:SAMPLE_LENGTH//2], spectrum[:SAMPLE_LENGTH//2])
        plt.title('Спектр зашумленного сигнала')
        plt.xlabel('Частота (Гц)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Основной цикл программы"""
        while True:
            self.print_menu()
            mod_choice = input(">> ").strip().lower()
            
            if mod_choice == 'q':
                break
            
            if mod_choice not in self.mod_types:
                print("Неверный выбор! Попробуйте снова.")
                continue
            
            self.print_snr_menu()
            snr_choice = input(">> ").strip()
            
            if snr_choice not in self.snr_levels:
                print("Неверный выбор! Попробуйте снова.")
                continue
            
            print("\nГенерация сигнала...")
            if self.generate_signal(mod_choice, snr_choice):
                while True:
                    show_plots = input("\nПоказать графики? (y/n) >> ").strip().lower()
                    if show_plots == 'y':
                        self.plot_signals()
                        break
                    elif show_plots == 'n':
                        break
                    else:
                        print("Пожалуйста, введите 'y' или 'n'")

def generate_modulated_signal(mod_type, snr):
    """Генерация модулированного сигнала для заданного типа модуляции и SNR
    
    Returns:
        tuple: (message, clean_signal, noisy_signal)
    """
    # Генерация временной оси
    t = np.linspace(0, SAMPLE_LENGTH/SAMPLING_RATE, SAMPLE_LENGTH)
    # Генерация модулирующего сигнала и модулированного сигнала
    carrier_freq = np.random.uniform(SAMPLING_RATE/20, SAMPLING_RATE/2/3)
    carrier = SignalGenerator.generate_carrier(t, carrier_freq)
    if mod_type in ['AM', 'FM', 'PM']:
        message = SignalGenerator.generate_analog_message(t, mode='random_sin')
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
        message = np.repeat(bits, len(t) // len(bits))
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
            message2 = np.repeat(bits2, len(t) // len(bits2))
            clean_signal = (message * carrier + message2 * np.cos(2 * np.pi * carrier_freq * t + np.pi/2)) / np.sqrt(2)
    # Добавление шума
    noisy_signal = add_noise(clean_signal, snr)
    return message, clean_signal, noisy_signal

if __name__ == "__main__":
    generator = ModulationGenerator()
    generator.run()